from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

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
    ai_result, risk_detected = await _analyze_photo(contents, ext)
    yolo_result = analyze_with_yolo(contents, ext)

    if yolo_result.get('available'):
        ai_result = f'{ai_result}\n\n[YOLO Detection] {yolo_result["summary"]}'
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
        db.add(
            Alert(
                level='high',
                message=f'[AI Photo Analysis] Risk detected - {ai_result[:80]}',
                source='Gemini+YOLO Vision',
            )
        )

    db.commit()
    db.refresh(photo)

    response = photo.__dict__.copy()
    response['yolo'] = yolo_result
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
        '4. Recommended Action: one short action for the manager\n'
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
