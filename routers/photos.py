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
    'helmet': '\uC548\uC804\uBAA8 \uC810\uAC80 \uD544\uC694',
    'vest': '\uC548\uC804\uC870\uB07C \uC810\uAC80 \uD544\uC694',
    'harness': '\uC548\uC804\uB300 \uC810\uAC80 \uD544\uC694',
    'gloves': '\uC7A5\uAC11 \uC810\uAC80 \uD544\uC694',
    'boots': '\uC548\uC804\uD654 \uC810\uAC80 \uD544\uC694',
    'goggles': '\uBCF4\uC548\uACBD \uC810\uAC80 \uD544\uC694',
    'mask': '\uB9C8\uC2A4\uD06C \uC810\uAC80 \uD544\uC694',
}


def _zone_label(zone: Zone | None, zone_id: int | None) -> str:
    if zone and zone.name:
        return zone.name
    if zone_id:
        return f'{chr(64 + zone_id)}\uAD6C\uC5ED' if 1 <= zone_id <= 26 else f'\uAD6C\uC5ED {zone_id}'
    return '\uAD6C\uC5ED \uBBF8\uC9C0\uC815'


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
    final_judgement = ''
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

        if lowered.startswith('5. final judgement:') or lowered.startswith('final judgement:'):
            final_judgement = line.split(':', 1)[1].strip() if ':' in line else line
            current_section = 'judgement'
            continue

        if lowered.startswith('risk:') or lowered.startswith('risk'):
            current_section = ''
            continue

        if current_section == 'hazard' and line.startswith('-'):
            value = line.lstrip('- ').strip()
            if value.lower() != 'none':
                hazard_points.append(value)
        elif current_section == 'scene':
            scene_summary = f'{scene_summary}\n{line}'.strip() if scene_summary else line
        elif current_section == 'ppe':
            ppe_check = f'{ppe_check}\n{line}'.strip() if ppe_check else line
        elif current_section == 'action':
            recommended_action = f'{recommended_action}\n{line}'.strip() if recommended_action else line
        elif current_section == 'judgement':
            final_judgement = f'{final_judgement}\n{line}'.strip() if final_judgement else line
        elif not scene_summary:
            scene_summary = line

    return {
        'scene_summary': scene_summary,
        'hazard_points': hazard_points,
        'ppe_check': ppe_check,
        'recommended_action': recommended_action,
        'final_judgement': final_judgement,
    }


def _risk_tags_from_text(text: str) -> list[str]:
    lowered = (text or '').lower()
    tags: list[str] = []
    if any(token in lowered for token in ('helmet', 'hardhat')):
        tags.append('\uC548\uC804\uBAA8 \uC810\uAC80 \uD544\uC694')
    if any(token in lowered for token in ('vest', 'hi-vis', 'reflective')):
        tags.append('\uC548\uC804\uC870\uB07C \uC810\uAC80 \uD544\uC694')
    if any(token in lowered for token in ('harness', 'lifeline')):
        tags.append('\uC548\uC804\uB300 \uC810\uAC80 \uD544\uC694')
    if any(token in lowered for token in ('fall', 'ladder', 'scaffold', 'edge', 'opening')):
        tags.append('\uB099\uC0C1 \uC704\uD5D8')
    if any(token in lowered for token in ('electric', 'wire', 'cable', 'spark', 'fire', 'flame', 'smoke')):
        tags.append('\uC804\uAE30\u00B7\uD654\uC7AC \uC704\uD5D8')
    if any(token in lowered for token in ('excavator', 'forklift', 'crane', 'truck', 'equipment', 'loader', 'backhoe')):
        tags.append('\uC911\uC7A5\uBE44 \uC811\uADFC \uC704\uD5D8')
    if any(token in lowered for token in ('walkway', 'blocked', 'path', 'stacked', 'passage')):
        tags.append('\uD1B5\uB85C \uC801\uCE58 \uC704\uD5D8')
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


def _build_yolo_context(yolo_result: dict[str, Any]) -> str:
    if not yolo_result.get('available'):
        return 'YOLO metadata unavailable.'

    warnings = yolo_result.get('warnings') or []
    ppe_missing = yolo_result.get('ppe_missing') or []
    actions = yolo_result.get('recommended_actions') or []

    lines = [
        f"- YOLO summary: {yolo_result.get('summary') or 'No summary'}",
        f"- YOLO risk level: {str(yolo_result.get('risk_level', 'safe')).upper()}",
    ]

    if warnings:
        lines.append('- YOLO warnings: ' + ' | '.join(str(item) for item in warnings[:3]))
    if ppe_missing:
        lines.append('- YOLO PPE missing: ' + ', '.join(str(item) for item in ppe_missing))
    if actions:
        lines.append('- YOLO recommended actions: ' + ' | '.join(str(item) for item in actions[:3]))

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

    scene_summary = parsed.get('scene_summary') or '\uC0AC\uC9C4 \uBD84\uC11D \uC694\uC57D\uC774 \uC5C6\uC2B5\uB2C8\uB2E4.'
    risk_types = _derive_risk_types(parsed, effective_yolo)
    recommended_actions = effective_yolo.get('recommended_actions') or []
    recommended_action = parsed.get('recommended_action') or (recommended_actions[0] if recommended_actions else '') or '\uD604\uC7A5 \uC0C1\uD0DC\uB97C \uD655\uC778\uD574 \uC8FC\uC138\uC694.'
    risk_level = _display_risk_level(bool(photo.risk_detected), effective_yolo)
    vision_source = _vision_source(gemini_text, effective_yolo)
    yolo_summary = _extract_yolo_summary(yolo_text, effective_yolo)
    hazard_points = parsed.get('hazard_points') or []
    ppe_check = parsed.get('ppe_check') or '\uBCF4\uD638\uAD6C \uCC29\uC6A9 \uC5EC\uBD80\uB97C \uD604\uC7A5\uC5D0\uC11C \uD55C \uBC88 \uB354 \uD655\uC778\uD574 \uC8FC\uC138\uC694.'
    final_judgement = parsed.get('final_judgement') or (
        '\uC989\uC2DC \uD655\uC778\uC774 \uD544\uC694\uD55C \uC704\uD5D8 \uC9D5\uD6C4\uAC00 \uC788\uC2B5\uB2C8\uB2E4.' if risk_level in {'high', 'caution'}
        else '\uD604\uC7AC \uAE30\uC900\uC73C\uB85C \uB69C\uB837\uD55C \uC704\uD5D8 \uC815\uD669\uC740 \uD655\uC778\uB418\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4.'
    )

    base['scene_summary'] = scene_summary
    base['risk_types'] = risk_types
    base['hazard_points'] = hazard_points
    base['ppe_check'] = ppe_check
    base['recommended_action'] = recommended_action
    base['final_judgement'] = final_judgement
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
    return f'[\uC0AC\uC9C4 \uBD84\uC11D] {zone_label} {risk_part} \u00B7 {recommended_action}'


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
    zone = db.query(Zone).filter(Zone.id == zone_id).first() if zone_id else None
    zone_label = _zone_label(zone, zone_id)
    yolo_result = analyze_with_yolo(contents, ext)
    gemini_result, gemini_risk = await _analyze_photo(contents, ext, zone_label, yolo_result)

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


async def _analyze_photo(image_bytes: bytes, ext: str, zone_label: str, yolo_result: dict[str, Any]) -> tuple[str, bool]:
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

    yolo_context = _build_yolo_context(yolo_result)
    prompt = (
        'You are a construction site safety manager assistant.\n'
        'Analyze the uploaded photo for a Korean construction site report.\n'
        f'The target zone is {zone_label}.\n'
        'Use the YOLO metadata below as supporting evidence when it is available.\n'
        'Do not guess details that are not visible.\n'
        'Write the CONTENT in Korean only, but keep the section headers below exactly in English so the server can parse them.\n'
        'Focus on practical safety judgement, not generic image description.\n'
        'Write the response as a structured field report that reads naturally to a site manager.\n'
        'Do not answer too briefly. Each section should contain enough detail to be useful in a real work log.\n'
        '\n'
        '[YOLO metadata]\n'
        f'{yolo_context}\n'
        '\n'
        'Required format:\n'
        '1. Scene Summary: write at least 2 paragraphs in Korean.\n'
        '- Paragraph 1 should describe the visible work situation, work environment, and major objects in 2-3 sentences.\n'
        '- Paragraph 2 should explain worker movement, equipment/material arrangement, and the overall work context in 2-3 sentences.\n'
        '2. Hazard Points:\n'
        '- write 2-4 bullet lines in Korean when hazards exist.\n'
        '- each bullet should include where the hazard is, why it is risky, and who may be affected.\n'
        '- write exactly "- 특이 위험요소 없음" only when no meaningful hazard is visible.\n'
        '3. PPE Check: write 2-3 Korean sentences about missing or properly worn PPE and whether immediate confirmation is needed.\n'
        '4. Recommended Action: write 3-4 Korean action sentences for the site manager, not just short keywords.\n'
        '5. Final Judgement: write 2-3 Korean sentences summarizing whether this scene requires immediate intervention, additional monitoring, or regular follow-up.\n'
        '6. Final line must be exactly RISK:YES or RISK:NO\n'
        'Mark RISK:YES if there is missing PPE, fall risk, equipment conflict, electrical/fire issue, blocked path, or any action requiring immediate check.'
    )

    try:
        result_text = analyze_image(prompt, image_bytes, mime_type, max_output_tokens=1800)
    except Exception as error:
        return f'AI analysis failed (Gemini Vision): {error}', False

    upper_text = result_text.upper()
    risk_detected = 'RISK:YES' in upper_text
    clean_text = result_text.replace('RISK:YES', '').replace('RISK:NO', '').strip()
    if not clean_text:
        clean_text = 'AI analysis completed, but no readable summary was returned.'
    return clean_text, risk_detected
