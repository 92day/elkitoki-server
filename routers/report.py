from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from database import get_db
from models.models import Report

router = APIRouter(prefix='/api/reports', tags=['reports'])


class ReportCreate(BaseModel):
    text_content: str = Field(..., min_length=1, max_length=4000)
    translated_text: str = Field(default='', max_length=4000)
    source_language: str = Field(default='ko', min_length=2, max_length=10)
    target_language: str = Field(default='vi', min_length=2, max_length=10)
    author_name: str = Field(default='Site Manager', max_length=50)

    @field_validator('text_content', mode='before')
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = (value or '').strip()
        if not cleaned:
            raise ValueError('text_content cannot be empty')
        return cleaned


@router.get('/')
def get_reports(db: Session = Depends(get_db)):
    return db.query(Report).order_by(Report.created_at.desc()).all()


@router.post('/')
def create_report(data: ReportCreate, db: Session = Depends(get_db)):
    report = Report(
        date=str(date.today()),
        text_content=data.text_content,
        translated_text=data.translated_text.strip() or None,
        source_language=data.source_language.strip().lower(),
        target_language=data.target_language.strip().lower(),
        author_name=data.author_name.strip() or 'Site Manager',
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@router.delete('/{report_id}')
def delete_report(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail='Report not found.')

    db.delete(report)
    db.commit()
    return {'ok': True}
