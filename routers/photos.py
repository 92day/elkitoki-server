"""
photos.py - field photo upload + Gemini Vision analysis
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import get_db
from gemini_client import analyze_image
from models.models import Alert, Photo

load_dotenv()
router = APIRouter(prefix="/api/photos", tags=["photos"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "../uploads")
PHOTOS_DIR = os.path.join(UPLOAD_DIR, "photos")
os.makedirs(PHOTOS_DIR, exist_ok=True)


@router.get("/")
def get_photos(zone_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(Photo)
    if zone_id:
        q = q.filter(Photo.zone_id == zone_id)
    return q.order_by(Photo.taken_at.desc()).all()


@router.post("/")
async def upload_photo(
    file: UploadFile = File(...),
    zone_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(PHOTOS_DIR, filename)

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")

    with open(filepath, "wb") as saved_file:
        saved_file.write(contents)

    ai_result, risk_detected = await _analyze_photo(contents, ext)

    photo = Photo(
        zone_id=zone_id,
        file_path=filepath,
        original_name=file.filename,
        ai_result=ai_result,
        risk_detected=risk_detected,
    )
    db.add(photo)

    if risk_detected:
        db.add(
            Alert(
                level="high",
                message=f"[AI 사진분석] 위험요소 감지 - {ai_result[:80]}",
                source="Gemini Vision",
            )
        )

    db.commit()
    db.refresh(photo)
    return photo


@router.get("/file/{photo_id}")
def get_photo_file(photo_id: int, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo or not os.path.exists(photo.file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(photo.file_path)


@router.delete("/{photo_id}")
def delete_photo(photo_id: int, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="사진을 찾을 수 없습니다.")

    file_path = photo.file_path
    db.delete(photo)
    db.commit()

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    return {"ok": True}


async def _analyze_photo(image_bytes: bytes, ext: str) -> tuple[str, bool]:
    """
    Return (analysis_text, risk_detected)
    """
    media_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime_type = media_map.get(ext.lower(), "image/jpeg")

    prompt = (
        "You are a construction site safety expert. Analyze the image and respond in Korean.\n"
        "Format:\n"
        "1. 현재 작업 상황 요약 (1~2문장)\n"
        "2. 감지된 안전 위험요소 (없으면 '없음')\n"
        "3. 안전장비 착용 여부\n"
        "4. 최종 판정: 위험 또는 정상\n"
        "The last line must be exactly RISK:YES or RISK:NO."
    )

    result_text = analyze_image(prompt, image_bytes, mime_type, max_output_tokens=500)
    risk_detected = "RISK:YES" in result_text.upper()
    clean_text = result_text.replace("RISK:YES", "").replace("RISK:NO", "").strip()
    return clean_text, risk_detected
