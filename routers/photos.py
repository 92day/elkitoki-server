from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env', override=False)

from database import get_db
from gemini_client import analyze_image, is_gemini_configured
from models.models import Alert, Photo
from yolo_client import analyze_with_yolo

router = APIRouter(prefix='/api/photos', tags=['photos'])

upload_dir_value = os.getenv('UPLOAD_DIR', './uploads')
UPLOAD_DIR = Path(upload_dir_value)
if not UPLOAD_DIR.is_absolute():
    UPLOAD_DIR = (BASE_DIR / UPLOAD_DIR).resolve()
PHOTOS_DIR = UPLOAD_DIR / 'photos'
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def _alert_level_from_risk(risk_level: str) -> str:
    normalized = (risk_level or '').strip().lower()
    if normalized == 'high':
        return 'high'
    if normalized == 'medium':
        return 'mid'
    if normalized == 'low':
        return 'low'
    if normalized == 'safe':
        return 'mid'
    return 'high'


def _extract_gemini_ppe_warning(gemini_result: str) -> str:
    for raw_line in gemini_result.splitlines():
        line = ' '.join(raw_line.strip().split())
        lowered = line.lower()
        if 'ppe check' not in lowered:
            continue
        if any(token in lowered for token in ('missing', 'without', 'not wearing', 'no helmet', 'no vest', 'no harness', 'no gloves', 'no boots')):
            return line
    return ''


def _extract_gemini_action(gemini_result: str) -> str:
    for raw_line in gemini_result.splitlines():
        line = ' '.join(raw_line.strip().split())
        lowered = line.lower()
        if lowered.startswith('4. recommended action') or lowered.startswith('recommended action'):
            return line
    return ''


def _build_yolo_block(yolo_result: dict[str, Any]) -> str:
    summary = yolo_result.get('summary') or 'No summary'
    risk_level = str(yolo_result.get('risk_level', 'safe')).upper()
    warnings = yolo_result.get('warnings') or []
    ppe_missing = yolo_result.get('ppe_missing') or []
    actions = yolo_result.get('recommended_actions') or []

    lines = [
        '[YOLO Safety Assessment]',
        f'Risk Level: {risk_level}',
        f'Detection Summary: {summary}',
    ]

    if warnings:
        lines.append('Warnings: ' + ' | '.join(str(w) for w in warnings))
    if ppe_missing:
        lines.append('PPE Warning: Missing ' + ', '.join(str(item) for item in ppe_missing))
    if actions:
        lines.append('Recommended Actions:')
        for idx, action in enumerate(actions[:3], start=1):
            lines.append(f'{idx}. {action}')

    return '\n'.join(lines)


def _build_photo_alert_message(
    ai_result: str,
    yolo_result: dict[str, Any],
    gemini_ppe_warning: str,
    gemini_action: str,
) -> str:
    warnings = yolo_result.get('warnings') or []
    ppe_missing = yolo_result.get('ppe_missing') or []
    actions = yolo_result.get('recommended_actions') or []

    parts: list[str] = ['[AI Photo Analysis]']
    if ppe_missing:
        parts.append('PPE missing: ' + ', '.join(str(item) for item in ppe_missing))
    elif gemini_ppe_warning:
        parts.append('PPE warning (Gemini): ' + gemini_ppe_warning)
    elif warnings:
        parts.append(str(warnings[0]))
    else:
        parts.append(ai_result[:100].replace('\n', ' '))

    if actions:
        parts.append('Action: ' + str(actions[0]))
    elif gemini_action:
        parts.append('Action: ' + gemini_action)
    else:
        parts.append('Action: Safety manager should perform immediate on-site check.')

    return ' | '.join(parts)


@router.get('/')
def get_photos(zone_id: Optional[int] = None, db: Session = Depends(get_db)):
    query = db.query(Photo)
    if zone_id:
        query = query.filter(Photo.zone_id == zone_id)
    return query.order_by(Photo.taken_at.desc()).all()


@router.post('/')
async def upload_photo(
    file: UploadFile = File(...),
    zone_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(file.filename or '')[1].lower() or '.jpg'
    filename = f'{uuid.uuid4().hex}{ext}'
    filepath = PHOTOS_DIR / filename

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail='Uploaded image file is empty.')

    filepath.write_bytes(contents)

    # Gemini 분석
    ai_result, risk_detected = await _analyze_photo(contents, ext)
    gemini_result = ai_result

    # YOLO 분석 추가
    yolo_result = analyze_with_yolo(contents, ext)
    yolo_block = _build_yolo_block(yolo_result)
    ai_result = f'{ai_result}\n\n{yolo_block}'
    if yolo_result.get('risk_detected'):
        risk_detected = True

    photo = Photo(
        zone_id=zone_id,
        file_path=str(filepath),
        original_name=file.filename,
        ai_result=ai_result,
        risk_detected=risk_detected,
    )
    db.add(photo)

    if risk_detected:
        yolo_risk_level = str(yolo_result.get('risk_level', 'safe')).lower()
        if yolo_risk_level in {'safe', 'low'}:
            yolo_risk_level = 'medium'
        gemini_ppe_warning = _extract_gemini_ppe_warning(gemini_result)
        gemini_action = _extract_gemini_action(gemini_result)
        alert_level = _alert_level_from_risk(yolo_risk_level)
        db.add(
            Alert(
                level=alert_level,
                message=_build_photo_alert_message(ai_result, yolo_result, gemini_ppe_warning, gemini_action),
                source='Gemini+YOLO Vision',
            )
        )

    db.commit()
    db.refresh(photo)

    response = photo.__dict__.copy()
    response['yolo'] = yolo_result
    response['risk_level'] = yolo_result.get('risk_level', 'safe')
    response['ppe_missing'] = yolo_result.get('ppe_missing', [])
    response['recommended_actions'] = yolo_result.get('recommended_actions', [])
    return response


@router.get('/file/{photo_id}')
def get_photo_file(photo_id: int, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo or not os.path.exists(photo.file_path):
        raise HTTPException(status_code=404, detail='Photo file not found.')
    return FileResponse(photo.file_path)


@router.delete('/{photo_id}')
def delete_photo(photo_id: int, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail='Photo not found.')

    file_path = photo.file_path
    db.delete(photo)
    db.commit()

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    return {'ok': True}


async def _analyze_photo(image_bytes: bytes, ext: str) -> tuple[str, bool]:
    media_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
    }
    mime_type = media_map.get(ext.lower(), 'image/jpeg')

    if not is_gemini_configured():
        return (
            'AI analysis unavailable. Configure GEMINI_API_KEY in the server .env file and restart the backend.',
            False,
        )

    prompt = (
        'You are a construction site safety inspector. Analyze the uploaded site photo and respond in English.\n'
        'Keep the answer concise and practical for a site manager.\n'
        'Required format:\n'
        '1. Scene Summary: one short sentence\n'
        '2. Hazard Points: bullet list with the exact dangerous area or object and why it is dangerous. Write "- none" if safe.\n'
        '3. PPE Check: whether helmets, vests, harnesses, gloves, or boots appear to be missing\n'
        '4. Recommended Action: up to 3 short, actionable instructions for the safety manager\n'
        '5. Final line must be exactly RISK:YES or RISK:NO\n'
        'If there is a fall risk, struck-by risk, heavy equipment conflict, fire/electrical issue, or missing PPE, mark RISK:YES.'
    )

    result_text = analyze_image(prompt, image_bytes, mime_type, max_output_tokens=500)
    upper_text = result_text.upper()
    risk_detected = 'RISK:YES' in upper_text
    clean_text = result_text.replace('RISK:YES', '').replace('RISK:NO', '').strip()
    if not clean_text:
        clean_text = 'AI analysis completed, but no readable summary was returned.'
    return clean_text, risk_detected
