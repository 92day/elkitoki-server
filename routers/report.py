import os
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from database import get_db
from gemini_client import analyze_text, is_gemini_configured
from models.models import DailySummary
from mongo_store import (
    DAILY_LOG_COLLECTIONS,
    clear_today_daily_log_entries,
    delete_document_by_id,
    fetch_today_daily_log_entries,
    fetch_today_report_entries,
    insert_report_entry,
)

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


def _is_worker_call(entry: dict) -> bool:
    return (entry.get('text_content') or '').startswith('[작업자 호출]') or (entry.get('text_content') or '').startswith('[작업자 요청]')


def _build_rule_based_summary(entries: list[dict]) -> tuple[str, int, str]:
    total_count = len(entries)
    translation_count = sum(1 for entry in entries if entry.get('entry_type') == 'translation')
    manual_count = sum(1 for entry in entries if entry.get('entry_type') == 'manual' and not _is_worker_call(entry))
    worker_call_count = sum(1 for entry in entries if _is_worker_call(entry))

    if total_count == 0:
        return (
            '오늘 저장된 소통 로그가 없습니다.\n\n기록이 생기면 이곳에서 요약을 생성할 수 있습니다.',
            0,
            'rule-based-summary',
        )

    content_lines: list[str] = []
    for entry in entries:
        text = (entry.get('text_content') or '').strip()
        if not text:
            continue
        if _is_worker_call(entry):
            content_lines.append(text.replace('[작업자 호출] ', '').replace('[작업자 요청] ', '').strip())
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


def _build_gemini_prompt(entries: list[dict]) -> str:
    lines: list[str] = []
    for index, entry in enumerate(entries, start=1):
        text = (entry.get('text_content') or '').strip()
        if not text:
            continue
        entry_label = '작업자 호출' if _is_worker_call(entry) else ('번역 기록' if entry.get('entry_type') == 'translation' else '수동 입력')
        created_at = entry.get('created_at') or ''
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


def _build_today_summary(entries: list[dict]) -> tuple[str, int, str]:
    if not entries:
        return _build_rule_based_summary(entries)

    if is_gemini_configured():
        prompt = _build_gemini_prompt(entries)
        summary_text = analyze_text(prompt, max_output_tokens=900).strip()
        if summary_text and not summary_text.startswith('AI analysis failed'):
            return summary_text, len(entries), _preferred_gemini_model_name()

    return _build_rule_based_summary(entries)


@router.get('/')
def get_reports():
    today = str(date.today())
    return fetch_today_report_entries(today)


@router.get('/today')
def get_today_reports():
    today = str(date.today())
    return fetch_today_report_entries(today)


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
    entries = list(reversed(fetch_today_report_entries(today)))
    summary_text, source_count, model_name = _build_today_summary(entries)
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
def create_report(data: ReportCreate):
    return insert_report_entry(
        entry_type=data.entry_type,
        text_content=data.text_content,
        translated_text=data.translated_text.strip() or None,
        source_language=data.source_language.strip().lower(),
        target_language=data.target_language.strip().lower(),
        author_name=data.author_name.strip() or 'Site Manager',
        date_text=str(date.today()),
    )


@router.delete('/daily-log/today')
def clear_today_daily_logs(db: Session = Depends(get_db)):
    today = str(date.today())
    clear_today_daily_log_entries(today)
    db.query(DailySummary).filter(DailySummary.summary_date == today).delete(synchronize_session=False)
    db.commit()
    return {'ok': True}


@router.delete('/{report_id}')
def delete_report(report_id: str):
    if ':' not in report_id:
        raise HTTPException(status_code=400, detail='Legacy MySQL report ids are no longer supported.')

    collection_name, mongo_id = report_id.split(':', 1)
    if collection_name not in DAILY_LOG_COLLECTIONS:
        raise HTTPException(status_code=400, detail='Unsupported log collection.')

    delete_document_by_id(collection_name, mongo_id)
    return {'ok': True}



def _entry_label(entry: dict) -> str:
    log_type = entry.get('log_type')
    if log_type == 'worker_call':
        return '작업자 호출'
    if log_type == 'alert':
        return '안전 알림'
    if log_type == 'translation':
        return '번역 기록'
    return '수동 입력'



def _build_rule_based_summary_from_entries(entries: list[dict]) -> tuple[str, int, str]:
    total_count = len(entries)
    translation_count = sum(1 for entry in entries if entry.get('log_type') == 'translation')
    manual_count = sum(1 for entry in entries if entry.get('log_type') == 'manual')
    worker_call_count = sum(1 for entry in entries if entry.get('log_type') == 'worker_call')
    alert_count = sum(1 for entry in entries if entry.get('log_type') == 'alert')

    if total_count == 0:
        return (
            '오늘 저장된 소통 로그가 없습니다.\n\n기록이 생기면 이곳에서 요약을 생성할 수 있습니다.',
            0,
            'rule-based-summary',
        )

    unique_lines: list[str] = []
    for entry in entries:
        text = (entry.get('text_content') or '').strip()
        if text and text not in unique_lines:
            unique_lines.append(text)

    highlights = unique_lines[:6]
    highlight_block = '\n'.join(f'- {item}' for item in highlights) if highlights else '- 주요 소통 기록이 없습니다.'

    summary_text = (
        f'오늘 총 {total_count}건의 기록이 저장되었습니다.\n'
        f'- 번역 기록 {translation_count}건\n'
        f'- 수동 입력 {manual_count}건\n'
        f'- 작업자 호출 {worker_call_count}건\n'
        f'- 안전 알림 {alert_count}건\n\n'
        f'주요 내용\n{highlight_block}'
    )
    return summary_text, total_count, 'rule-based-summary'



def _build_gemini_prompt_from_entries(entries: list[dict]) -> str:
    lines: list[str] = []
    for index, entry in enumerate(entries, start=1):
        text = (entry.get('text_content') or '').strip()
        if not text:
            continue
        entry_label = _entry_label(entry)
        created_at = entry.get('created_at') or ''
        lines.append(f'{index}. [{entry_label}] [{created_at}] {text}')

    joined_logs = '\n'.join(lines) if lines else '기록 없음'
    return (
        '당신은 건설 현장 관리자의 작업일지를 정리하는 한국어 비서입니다.\n'
        '아래 오늘의 소통 로그와 안전 알림 로그를 읽고, 중복 없이 자연스럽고 간결한 한국어로 요약하세요.\n'
        '반드시 다음 4개 제목으로만 작성하세요.\n'
        '1. 오늘의 핵심 작업\n'
        '2. 주요 지시 및 소통\n'
        '3. 위험 및 주의 사항\n'
        '4. 후속 조치\n\n'
        '주의사항:\n'
        '- 로그에 없는 내용은 추측하지 마세요.\n'
        '- 너무 장황하지 않게 8~12문장 내로 정리하세요.\n'
        '- 작업자 호출과 안전 알림이 있으면 필요한 경우 주요 지시 및 소통 또는 위험 및 주의 사항에 반영하세요.\n\n'
        f'오늘의 로그:\n{joined_logs}'
    )



def _build_today_summary_from_entries(entries: list[dict]) -> tuple[str, int, str]:
    if not entries:
        return _build_rule_based_summary_from_entries(entries)

    if is_gemini_configured():
        prompt = _build_gemini_prompt_from_entries(entries)
        summary_text = analyze_text(prompt, max_output_tokens=900).strip()
        if summary_text and not summary_text.startswith('AI analysis failed'):
            return summary_text, len(entries), _preferred_gemini_model_name()

    return _build_rule_based_summary_from_entries(entries)


@router.get('/daily-log/today')
def get_today_daily_log_entries():
    today = str(date.today())
    return fetch_today_daily_log_entries(today)


@router.get('/daily-log/summary/today')
def get_today_daily_log_summary(db: Session = Depends(get_db)):
    today = str(date.today())
    summary = (
        db.query(DailySummary)
        .filter(DailySummary.summary_date == today)
        .order_by(DailySummary.updated_at.desc(), DailySummary.created_at.desc())
        .first()
    )
    return summary


@router.post('/daily-log/summary/today/generate')
def generate_today_daily_log_summary(db: Session = Depends(get_db)):
    today = str(date.today())
    entries = fetch_today_daily_log_entries(today)
    summary_text, source_count, model_name = _build_today_summary_from_entries(entries)

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
