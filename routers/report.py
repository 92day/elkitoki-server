"""
report.py
- POST /api/reports/: audio upload -> OpenAI Whisper STT -> Gemini analysis -> save
- POST /api/reports/text: manual text -> Gemini analysis -> save
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import os
import uuid

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from openai import OpenAI
from sqlalchemy.orm import Session

from database import get_db
from gemini_client import analyze_text
from models.models import Report

load_dotenv()
router = APIRouter(prefix="/api/reports", tags=["reports"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "../uploads"))
AUDIO_DIR = UPLOAD_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

openai_client: OpenAI | None = None
if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")


@router.get("/")
def get_reports(db: Session = Depends(get_db)):
    return db.query(Report).order_by(Report.created_at.desc()).all()


@router.post("/")
async def create_report_from_audio(
    file: UploadFile = File(...),
    author_name: str = Form("관리자"),
    db: Session = Depends(get_db),
):
    ext = Path(file.filename or "").suffix.lower() or ".webm"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = AUDIO_DIR / filename

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="오디오 파일이 비어 있습니다.")

    filepath.write_bytes(contents)
    text_content = _transcribe_audio(filepath, file.filename or filename)
    ai_analysis = _analyze_report_with_gemini(text_content)

    report = Report(
        date=str(date.today()),
        text_content=text_content,
        ai_analysis=ai_analysis,
        author_name=author_name or "관리자",
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@router.post("/text")
async def create_report_from_text(
    text_content: str = Form(...),
    author_name: str = Form("관리자"),
    db: Session = Depends(get_db),
):
    text_value = text_content.strip()
    if not text_value:
        raise HTTPException(status_code=400, detail="보고서 텍스트가 비어 있습니다.")

    ai_analysis = _analyze_report_with_gemini(text_value)
    report = Report(
        date=str(date.today()),
        text_content=text_value,
        ai_analysis=ai_analysis,
        author_name=author_name or "관리자",
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@router.delete("/{report_id}")
def delete_report(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="보고서를 찾을 수 없습니다.")
    db.delete(report)
    db.commit()
    return {"ok": True}


def _transcribe_audio(audio_path: Path, original_filename: str) -> str:
    if openai_client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY가 설정되지 않았습니다.")

    try:
        with audio_path.open("rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model=STT_MODEL,
                file=(original_filename, audio_file),
            )
        text = (transcript.text or "").strip()
        if not text:
            return "음성 변환 결과가 비어 있습니다."
        return text
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"STT 처리 실패: {exc}") from exc


def _analyze_report_with_gemini(text: str) -> str:
    prompt = (
        "You are a construction site safety expert.\n"
        "Analyze the daily report below and respond in Korean.\n\n"
        f"[Report]\n{text}\n\n"
        "Format:\n"
        "1. 핵심 요약 (1~2문장)\n"
        "2. 감지된 위험요소 (없으면 '없음')\n"
        "3. 즉시 조치 필요 사항\n"
        "4. 위험도: 낮음 / 보통 / 높음"
    )
    return analyze_text(prompt, max_output_tokens=800)
