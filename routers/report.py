import os
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from database import get_db
from gemini_client import analyze_text, is_gemini_configured
from models.models import DailySummary, Report

router = APIRouter(prefix='/api/reports', tags=['reports'])

REPORT_ENTRY_TYPES = ['translation', 'manual']


class ReportCreate(BaseModel):
    text_content: str = Field(..., min_length=1, max_length=4000)
    translated_text: str = Field(default='', max_length=4000)
    source_language: str = Field(default='ko', min_length=2, max_length=10)
    target_language: str = Field(default='vi', min_length=2, max_length=10)
    author_name: str = Field(default='Site Manager', max_length=50)
    entry_type: str = Field(default='translation', min_length=4, max_length=20)

    @field_validator('text_content', mode='before')
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = (value or '').strip()
        if not cleaned:
            raise ValueError('text_content cannot be empty')
        return cleaned

    @field_validator('entry_type', mode='before')
    @classmethod
    def validate_entry_type(cls, value: str) -> str:
        cleaned = (value or 'translation').strip().lower()
        if cleaned not in REPORT_ENTRY_TYPES:
            raise ValueError('entry_type must be translation or manual')
        return cleaned


class DailySummaryUpsert(BaseModel):
    summary_text: str = Field(..., min_length=1, max_length=8000)
    source_count: int = Field(default=0, ge=0)
    model_name: Optional[str] = Field(default=None, max_length=50)

    @field_validator('summary_text', mode='before')
    @classmethod
    def validate_summary_text(cls, value: str) -> str:
        cleaned = (value or '').strip()
        if not cleaned:
            raise ValueError('summary_text cannot be empty')
        return cleaned


def _is_worker_call(report: Report) -> bool:
    return (report.text_content or '').startswith('[작업자 호출]')


def _build_rule_based_summary(reports: list[Report]) -> tuple[str, int, str]:
    total_count = len(reports)
    translation_count = sum(1 for report in reports if report.entry_type == 'translation')
    manual_count = sum(1 for report in reports if report.entry_type == 'manual' and not _is_worker_call(report))
    worker_call_count = sum(1 for report in reports if _is_worker_call(report))

    if total_count == 0:
        return (
            '오늘 저장된 소통 로그가 없습니다.\n\n기록이 생기면 이곳에서 요약을 생성할 수 있습니다.',
            0,
            'rule-based-summary',
        )

    content_lines = []
    for report in reports:
        text = (report.text_content or '').strip()
        if not text:
            continue
        if _is_worker_call(report):
            content_lines.append(text.replace('[작업자 호출] ', '').strip())
        else:
            content_lines.append(text)

    unique_lines: list[str] = []
    for text in content_lines:
        if text not in unique_lines:
            unique_lines.append(text)

    highlights = unique_lines[:5]
    highlight_block = '\n'.join(f'- {item}' for item in highlights) if highlights else '- 주요 소통 기록이 없습니다.'

    summary_text = (
        f'오늘 총 {total_count}건의 소통이 기록되었습니다.\n'
        f'- 번역 기록 {translation_count}건\n'
        f'- 수동 입력 {manual_count}건\n'
        f'- 작업자 호출 {worker_call_count}건\n\n'
        f'주요 내용\n{highlight_block}'
    )
    return summary_text, total_count, 'rule-based-summary'


def _build_gemini_prompt(reports: list[Report]) -> str:
    lines: list[str] = []
    for index, report in enumerate(reports, start=1):
        text = (report.text_content or '').strip()
        if not text:
            continue
        entry_label = '작업자 호출' if _is_worker_call(report) else ('번역 기록' if report.entry_type == 'translation' else '수동 입력')
        created_at = report.created_at.isoformat() if report.created_at else ''
        lines.append(f'{index}. [{entry_label}] [{created_at}] {text}')

    joined_logs = '\n'.join(lines) if lines else '기록 없음'
    return (
        '당신은 건설 현장 관리자의 작업일지를 정리하는 한국어 비서입니다.\n'
        '아래 오늘의 소통 로그를 읽고, 중복 없이 자연스럽고 간결한 한국어로 요약하세요.\n'
        '반드시 다음 4개 제목으로만 작성하세요.\n'
        '1. 오늘의 핵심 작업\n'
        '2. 주요 지시 및 소통\n'
        '3. 위험 및 주의 사항\n'
        '4. 후속 조치\n\n'
        '주의사항:\n'
        '- 로그에 없는 내용은 추측하지 마세요.\n'
        '- 너무 장황하지 않게 8~12문장 내로 정리하세요.\n'
        '- 작업자 호출이 있으면 필요한 경우 주요 지시 및 소통 또는 후속 조치에 녹여 쓰세요.\n\n'
        f'오늘의 소통 로그:\n{joined_logs}'
    )


def _preferred_gemini_model_name() -> str:
    raw = (os.getenv('GEMINI_TEXT_MODELS') or '').strip()
    if raw:
        return raw.split(',')[0].strip() or 'gemini-text'
    return 'gemini-2.5-flash'


def _build_today_summary(reports: list[Report]) -> tuple[str, int, str]:
    if not reports:
        return _build_rule_based_summary(reports)

    if is_gemini_configured():
        prompt = _build_gemini_prompt(reports)
        summary_text = analyze_text(prompt, max_output_tokens=900).strip()
        if summary_text and not summary_text.startswith('AI analysis failed'):
            return summary_text, len(reports), _preferred_gemini_model_name()

    return _build_rule_based_summary(reports)


@router.get('/')
def get_reports(db: Session = Depends(get_db)):
    return db.query(Report).order_by(Report.created_at.desc()).all()


@router.get('/today')
def get_today_reports(db: Session = Depends(get_db)):
    today = str(date.today())
    return (
        db.query(Report)
        .filter(Report.date == today)
        .order_by(Report.created_at.desc())
        .all()
    )


@router.get('/summary/today')
def get_today_summary(db: Session = Depends(get_db)):
    today = str(date.today())
    summary = (
        db.query(DailySummary)
        .filter(DailySummary.summary_date == today)
        .order_by(DailySummary.updated_at.desc(), DailySummary.created_at.desc())
        .first()
    )
    return summary


@router.put('/summary/today')
def upsert_today_summary(data: DailySummaryUpsert, db: Session = Depends(get_db)):
    today = str(date.today())
    summary = (
        db.query(DailySummary)
        .filter(DailySummary.summary_date == today)
        .order_by(DailySummary.updated_at.desc(), DailySummary.created_at.desc())
        .first()
    )

    if summary is None:
        summary = DailySummary(summary_date=today)
        db.add(summary)

    summary.summary_text = data.summary_text
    summary.source_count = data.source_count
    summary.model_name = data.model_name.strip() if data.model_name else None

    db.commit()
    db.refresh(summary)
    return summary


@router.post('/summary/today/generate')
def generate_today_summary(db: Session = Depends(get_db)):
    today = str(date.today())
    reports = (
        db.query(Report)
        .filter(Report.date == today)
        .order_by(Report.created_at.asc())
        .all()
    )

    summary_text, source_count, model_name = _build_today_summary(reports)
    summary = (
        db.query(DailySummary)
        .filter(DailySummary.summary_date == today)
        .order_by(DailySummary.updated_at.desc(), DailySummary.created_at.desc())
        .first()
    )

    if summary is None:
        summary = DailySummary(summary_date=today)
        db.add(summary)

    summary.summary_text = summary_text
    summary.source_count = source_count
    summary.model_name = model_name

    db.commit()
    db.refresh(summary)
    return summary


@router.post('/')
def create_report(data: ReportCreate, db: Session = Depends(get_db)):
    report = Report(
        date=str(date.today()),
        entry_type=data.entry_type,
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
