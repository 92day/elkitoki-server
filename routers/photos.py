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
from models.models import Alert, Photo, Zone
from mongo_store import sync_alert_log
from yolo_client import analyze_with_yolo

router = APIRouter(prefix='/api/photos', tags=['photos'])

upload_dir_value = os.getenv('UPLOAD_DIR', './uploads')
UPLOAD_DIR = Path(upload_dir_value)
if not UPLOAD_DIR.is_absolute():
    UPLOAD_DIR = (BASE_DIR / UPLOAD_DIR).resolve()
PHOTOS_DIR = UPLOAD_DIR / 'photos'
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


RISK_TAG_LABELS = {
    'helmet': '안전모 점검 필요',
    'vest': '안전조끼 점검 필요',
    'harness': '안전대 점검 필요',
    'gloves': '장갑 점검 필요',
    'boots': '안전화 점검 필요',
    'goggles': '보안경 점검 필요',
    'mask': '마스크 점검 필요',
}


def _zone_label(zone: Zone | None, zone_id: int | None) -> str:
    if zone and zone.name:
        return zone.name
    if zone_id:
        return f'{chr(64 + zone_id)}구역' if 1 <= zone_id <= 26 else f'구역 {zone_id}'
    return '구역 미지정'


def _split_analysis_blocks(ai_result: str) -> tuple[str, str]:
    text = (ai_result or '').strip()
    yolo_markers = ['\n\n[YOLO Safety Assessment]\n', '\n\n[YOLO Detection] ']
    for marker in yolo_markers:
        if marker in text:
            gemini_text, yolo_text = text.split(marker, 1)
            if marker.strip() == '[YOLO Detection]':
                yolo_text = '[YOLO Detection] ' + yolo_text
            else:
                yolo_text = '[YOLO Safety Assessment]\n' + yolo_text
            return gemini_text.strip(), yolo_text.strip()
    return text, ''


def _parse_gemini_sections(gemini_text: str) -> dict[str, Any]:
    scene_summary = ''
    hazard_points: list[str] = []
    ppe_check = ''
    recommended_action = ''
    current_section = ''

    for raw_line in gemini_text.splitlines():
        line = ' '.join(raw_line.strip().split())
        if not line:
            continue
        lowered = line.lower()

        if lowered.startswith('1. scene summary:') or lowered.startswith('scene summary:'):
            scene_summary = line.split(':', 1)[1].strip() if ':' in line else line
            current_section = 'scene'
            continue

        if lowered.startswith('2. hazard points:') or lowered.startswith('hazard points:'):
            current_section = 'hazard'
            remainder = line.split(':', 1)[1].strip() if ':' in line else ''
            if remainder and remainder.lower() != 'none' and remainder != '- none':
                hazard_points.append(remainder.lstrip('- ').strip())
            continue

        if lowered.startswith('3. ppe check:') or lowered.startswith('ppe check:'):
            ppe_check = line.split(':', 1)[1].strip() if ':' in line else line
            current_section = 'ppe'
            continue

        if lowered.startswith('4. recommended action:') or lowered.startswith('recommended action:'):
            recommended_action = line.split(':', 1)[1].strip() if ':' in line else line
            current_section = 'action'
            continue

        if lowered.startswith('risk:') or lowered.startswith('risk'):
            current_section = ''
            continue

        if current_section == 'hazard' and line.startswith('-'):
            value = line.lstrip('- ').strip()
            if value.lower() != 'none':
                hazard_points.append(value)
        elif current_section == 'action' and not recommended_action:
            recommended_action = line
        elif not scene_summary:
            scene_summary = line

    return {
        'scene_summary': scene_summary,
        'hazard_points': hazard_points,
        'ppe_check': ppe_check,
        'recommended_action': recommended_action,
    }


def _risk_tags_from_text(text: str) -> list[str]:
    lowered = (text or '').lower()
    tags: list[str] = []
    if any(token in lowered for token in ('helmet', 'hardhat')):
        tags.append('안전모 점검 필요')
    if any(token in lowered for token in ('vest', 'hi-vis', 'reflective')):
        tags.append('안전조끼 점검 필요')
    if any(token in lowered for token in ('harness', 'lifeline')):
        tags.append('안전대 점검 필요')
    if any(token in lowered for token in ('fall', 'ladder', 'scaffold', 'edge', 'opening')):
        tags.append('낙상 위험')
    if any(token in lowered for token in ('electric', 'wire', 'cable', 'spark', 'fire', 'flame', 'smoke')):
        tags.append('전기·화재 위험')
    if any(token in lowered for token in ('excavator', 'forklift', 'crane', 'truck', 'equipment', 'loader', 'backhoe')):
        tags.append('중장비 접근 위험')
    if any(token in lowered for token in ('walkway', 'blocked', 'path', 'stacked', 'passage')):
        tags.append('통로 적치 위험')
    return tags


def _derive_risk_types(gemini_sections: dict[str, Any], yolo_result: dict[str, Any]) -> list[str]:
    risk_types: list[str] = []

    for item in yolo_result.get('ppe_missing') or []:
        lowered = str(item).lower()
        for key, label in RISK_TAG_LABELS.items():
            if key in lowered and label not in risk_types:
                risk_types.append(label)

    for warning in yolo_result.get('warnings') or []:
        for tag in _risk_tags_from_text(str(warning)):
            if tag not in risk_types:
                risk_types.append(tag)

    for bullet in gemini_sections.get('hazard_points') or []:
        for tag in _risk_tags_from_text(str(bullet)):
            if tag not in risk_types:
                risk_types.append(tag)

    ppe_check = gemini_sections.get('ppe_check') or ''
    for tag in _risk_tags_from_text(ppe_check):
        if tag not in risk_types:
            risk_types.append(tag)

    return risk_types[:5]


def _display_risk_level(risk_detected: bool, yolo_result: dict[str, Any]) -> str:
    yolo_level = str(yolo_result.get('risk_level', '')).lower()
    if yolo_level == 'high':
        return 'high'
    if yolo_level in {'medium', 'low'}:
        return 'caution'
    return 'caution' if risk_detected else 'safe'


def _vision_source(gemini_text: str, yolo_result: dict[str, Any]) -> str:
    has_gemini = bool((gemini_text or '').strip()) and 'AI analysis unavailable' not in gemini_text and 'AI analysis failed' not in gemini_text
    has_yolo = bool(yolo_result.get('available'))
    if has_gemini and has_yolo:
        return 'Gemini+YOLO'
    if has_yolo:
        return 'YOLO'
    return 'Gemini'


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
        lines.append('Warnings: ' + ' | '.join(str(item) for item in warnings))
    if ppe_missing:
        lines.append('PPE Warning: Missing ' + ', '.join(str(item) for item in ppe_missing))
    if actions:
        lines.append('Recommended Actions: ' + ' | '.join(str(item) for item in actions[:3]))
    return '\n'.join(lines)


def _extract_yolo_summary(yolo_text: str, yolo_result: dict[str, Any]) -> str:
    if yolo_result.get('summary'):
        return str(yolo_result['summary'])
    for line in yolo_text.splitlines():
        line = line.strip()
        if line.lower().startswith('detection summary:'):
            return line.split(':', 1)[1].strip()
        if line.startswith('[YOLO Detection]'):
            return line.replace('[YOLO Detection]', '').strip()
    return ''


def _structured_photo_response(photo: Photo, yolo_result: dict[str, Any] | None = None, zone: Zone | None = None) -> dict[str, Any]:
    base = photo.__dict__.copy()
    base.pop('_sa_instance_state', None)

    gemini_text, yolo_text = _split_analysis_blocks(photo.ai_result or '')
    parsed = _parse_gemini_sections(gemini_text)
    effective_yolo = yolo_result or {}

    scene_summary = parsed.get('scene_summary') or '사진 분석 요약이 없습니다.'
    risk_types = _derive_risk_types(parsed, effective_yolo)
    recommended_actions = effective_yolo.get('recommended_actions') or []
    recommended_action = parsed.get('recommended_action') or (recommended_actions[0] if recommended_actions else '') or '현장 상태를 확인해 주세요.'
    risk_level = _display_risk_level(bool(photo.risk_detected), effective_yolo)
    vision_source = _vision_source(gemini_text, effective_yolo)
    yolo_summary = _extract_yolo_summary(yolo_text, effective_yolo)

    base['scene_summary'] = scene_summary
    base['risk_types'] = risk_types
    base['recommended_action'] = recommended_action
    base['risk_level'] = risk_level
    base['vision_source'] = vision_source
    base['yolo_summary'] = yolo_summary
    base['ppe_missing'] = effective_yolo.get('ppe_missing', [])
    base['recommended_actions'] = recommended_actions
    base['zone_label'] = _zone_label(zone or getattr(photo, 'zone', None), photo.zone_id)
    if yolo_result is not None:
        base['yolo'] = effective_yolo
    return base


def _alert_level_from_display(display_level: str) -> str:
    if display_level == 'high':
        return 'high'
    if display_level == 'caution':
        return 'mid'
    return 'low'


def _build_photo_alert_message(zone_label: str, risk_types: list[str], recommended_action: str, scene_summary: str) -> str:
    risk_part = ', '.join(risk_types[:2]) if risk_types else scene_summary
    return f'[사진 분석] {zone_label} {risk_part} · {recommended_action}'


@router.get('/')
def get_photos(zone_id: Optional[int] = None, db: Session = Depends(get_db)):
    query = db.query(Photo)
    if zone_id:
        query = query.filter(Photo.zone_id == zone_id)
    photos = query.order_by(Photo.taken_at.desc()).all()
    return [_structured_photo_response(photo) for photo in photos]


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
    gemini_result, gemini_risk = await _analyze_photo(contents, ext)
    yolo_result = analyze_with_yolo(contents, ext)

    ai_result = gemini_result
    if yolo_result.get('available'):
        ai_result = f'{gemini_result}\n\n{_build_yolo_block(yolo_result)}'

    photo = Photo(
        zone_id=zone_id,
        file_path=str(filepath),
        original_name=file.filename,
        ai_result=ai_result,
        risk_detected=bool(gemini_risk or yolo_result.get('risk_detected')),
    )
    db.add(photo)
    db.flush()

    zone = db.query(Zone).filter(Zone.id == zone_id).first() if zone_id else None
    payload = _structured_photo_response(photo, yolo_result=yolo_result, zone=zone)

    if payload['risk_level'] != 'safe':
        alert = Alert(
            level=_alert_level_from_display(payload['risk_level']),
            message=_build_photo_alert_message(payload['zone_label'], payload['risk_types'], payload['recommended_action'], payload['scene_summary']),
            source=payload['vision_source'],
            zone_id=zone_id,
            zone_name=payload['zone_label'] if zone_id else None,
        )
        db.add(alert)
        db.flush()
        sync_alert_log(alert, 'photo_analysis')

    db.commit()
    db.refresh(photo)
    return _structured_photo_response(photo, yolo_result=yolo_result, zone=zone)


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

    try:
        result_text = analyze_image(prompt, image_bytes, mime_type, max_output_tokens=700)
    except Exception as error:
        return f'AI analysis failed (Gemini Vision): {error}', False

    upper_text = result_text.upper()
    risk_detected = 'RISK:YES' in upper_text
    clean_text = result_text.replace('RISK:YES', '').replace('RISK:NO', '').strip()
    if not clean_text:
        clean_text = 'AI analysis completed, but no readable summary was returned.'
    return clean_text, risk_detected
