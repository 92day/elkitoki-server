from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import get_db
from gemini_client import analyze_image
from models.models import Alert, Photo

load_dotenv()
router = APIRouter(prefix='/api/photos', tags=['photos'])

BASE_DIR = Path(__file__).resolve().parent.parent
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
                source='Gemini Vision',
            )
        )

    db.commit()
    db.refresh(photo)
    return photo


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

    prompt = (
        'You are a construction site safety expert. Analyze the image and respond in English.\n'
        'Format:\n'
        '1. Current site activity summary (1-2 sentences)\n'
        '2. Detected safety risks (write none if there are no issues)\n'
        '3. Whether safety equipment is being worn\n'
        '4. Final decision: risk or normal\n'
        'The last line must be exactly RISK:YES or RISK:NO.'
    )

    result_text = analyze_image(prompt, image_bytes, mime_type, max_output_tokens=500)
    risk_detected = 'RISK:YES' in result_text.upper()
    clean_text = result_text.replace('RISK:YES', '').replace('RISK:NO', '').strip()
    return clean_text, risk_detected
