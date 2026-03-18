import os
import re
from collections import Counter
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from database import get_db
from gemini_client import analyze_text_with_meta, is_gemini_configured
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
        summary_text = analyze_text(prompt, max_output_tokens=1800).strip()
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


def _clean_summary_text(text: str) -> str:
    cleaned = re.sub(r'\s+', ' ', (text or '').strip())
    cleaned = cleaned.replace('[작업자 호출] ', '').replace('[작업자 요청] ', '')
    return cleaned.strip()


def _extract_sentences(text: str) -> list[str]:
    normalized = re.sub(r'\s+', ' ', (text or '').strip())
    if not normalized:
        return []
    parts = re.split(r'(?<=[.!?。！？])\s+|(?<=다\.)\s+|(?<=요\.)\s+|\n+', normalized)
    return [part.strip() for part in parts if part and part.strip()]


def _shorten_for_summary(text: str, *, max_sentences: int = 3, max_chars: int = 220) -> str:
    cleaned = _clean_summary_text(text)
    if len(cleaned) <= max_chars:
        return cleaned

    sentences = _extract_sentences(cleaned)
    if not sentences:
        return cleaned[:max_chars].rstrip() + '...'

    shortened = ' '.join(sentences[:max_sentences]).strip()
    if len(shortened) > max_chars:
        return shortened[:max_chars].rstrip() + '...'
    return shortened


def _format_entry_time(entry: dict) -> str:
    created_at = str(entry.get('created_at') or '').strip()
    if not created_at:
        return '--:--'
    try:
        parsed = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        return parsed.astimezone().strftime('%H:%M')
    except ValueError:
        return '--:--'


def _build_summary_context(entries: list[dict]) -> dict:
    ordered_entries = sorted(entries, key=lambda entry: entry.get('created_at') or '')
    counts = Counter((entry.get('log_type') or 'manual') for entry in entries)
    worker_counter: Counter[str] = Counter()
    alert_counter: Counter[str] = Counter()
    alert_events: Counter[str] = Counter()
    translation_lines: list[str] = []
    manual_lines: list[str] = []
    manual_raw_texts: list[str] = []

    for entry in ordered_entries:
        log_type = entry.get('log_type')
        text = _clean_summary_text(entry.get('text_content') or '')
        if not text:
            continue

        time_label = _format_entry_time(entry)
        if log_type == 'translation':
            translation_lines.append(f'- {time_label} {text}')
            continue

        if log_type == 'worker_call':
            worker_counter[text] += 1
            continue

        if log_type == 'alert':
            alert_counter[text] += 1
            event_type = (entry.get('event_type') or '').strip()
            if event_type:
                alert_events[event_type] += 1
            continue

        manual_raw_texts.append(text)
        manual_lines.append(f'- {time_label} {_shorten_for_summary(text, max_sentences=4, max_chars=280)}')

    worker_lines = [
        f'- {label} ({count}회)'
        for label, count in worker_counter.most_common()
    ]
    alert_lines = [
        f'- {label} ({count}회)'
        for label, count in alert_counter.most_common()
    ]

    metadata_lines = [
        f'- 총 로그 {len(entries)}건',
        f"- 번역 기록 {counts.get('translation', 0)}건",
        f"- 작업자 호출/요청 {counts.get('worker_call', 0)}건",
        f"- 안전 알림 {counts.get('alert', 0)}건",
        f"- 수동 입력 {counts.get('manual', 0)}건",
    ]

    if alert_events.get('fall_detected'):
        metadata_lines.append(f"- 낙상 감지 {alert_events['fall_detected']}건")
    if alert_events.get('noise_abnormal'):
        metadata_lines.append(f"- 소음 경고 {alert_events['noise_abnormal']}건")

    manual_summary_lines = [
        f'- {_shorten_for_summary(text, max_sentences=3, max_chars=170)}'
        for text in manual_raw_texts[:5]
    ]

    log_analysis_lines = [
        f"- 총 로그 {len(entries)}건 중 번역 {counts.get('translation', 0)}건, 호출/요청 {counts.get('worker_call', 0)}건, 알림 {counts.get('alert', 0)}건, 수동 입력 {counts.get('manual', 0)}건",
    ]
    if worker_lines:
        log_analysis_lines.append(f"- 호출/요청은 {worker_lines[0].replace('- ', '', 1)}를 중심으로 이루어짐")
    if alert_lines:
        log_analysis_lines.append(f"- 주요 위험 신호는 {alert_lines[0].replace('- ', '', 1)}")
    if translation_lines:
        log_analysis_lines.append(f"- 번역 지시는 {translation_lines[0].replace('- ', '', 1)}를 포함해 현장 작업 전달에 사용됨")

    return {
        'counts': counts,
        'metadata_lines': metadata_lines,
        'log_analysis_lines': log_analysis_lines,
        'translation_lines': translation_lines[:10],
        'worker_lines': worker_lines[:8],
        'alert_lines': alert_lines[:8],
        'manual_lines': manual_lines[:8],
        'manual_summary_lines': manual_summary_lines,
    }


def _build_rule_based_summary_from_entries(entries: list[dict]) -> tuple[str, int, str]:
    if not entries:
        return (
            '오늘 저장된 소통 로그가 없습니다.\n\n기록이 생기면 이곳에서 요약을 생성할 수 있습니다.',
            0,
            'rule-based-summary',
        )

    context = _build_summary_context(entries)
    total_count = len(entries)
    translation_count = context['counts'].get('translation', 0)
    manual_count = context['counts'].get('manual', 0)
    worker_call_count = context['counts'].get('worker_call', 0)
    alert_count = context['counts'].get('alert', 0)

    highlights: list[str] = []
    if context['manual_lines']:
        highlights.append(context['manual_lines'][0].replace('- ', '', 1))
    if context['translation_lines']:
        highlights.append(context['translation_lines'][0].replace('- ', '', 1))
    if context['worker_lines']:
        highlights.append(context['worker_lines'][0].replace('- ', '', 1))
    if context['alert_lines']:
        highlights.append(context['alert_lines'][0].replace('- ', '', 1))

    highlight_block = '\n'.join(f'- {item}' for item in highlights[:6]) if highlights else '- 주요 소통 기록이 없습니다.'

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
    context = _build_summary_context(entries)

    metadata_block = '\n'.join(context['metadata_lines']) if context['metadata_lines'] else '- 로그 메타데이터 없음'
    log_analysis_block = '\n'.join(context['log_analysis_lines']) if context['log_analysis_lines'] else '- 로그 분석 없음'
    translation_block = '\n'.join(context['translation_lines']) if context['translation_lines'] else '- 번역 기록 없음'
    worker_block = '\n'.join(context['worker_lines']) if context['worker_lines'] else '- 작업자 호출/요청 없음'
    alert_block = '\n'.join(context['alert_lines']) if context['alert_lines'] else '- 안전 알림 없음'
    manual_block = '\n'.join(context['manual_lines']) if context['manual_lines'] else '- 수동 입력 없음'
    manual_summary_block = '\n'.join(context['manual_summary_lines']) if context['manual_summary_lines'] else '- 수동 입력 핵심 요약 없음'

    return (
        '당신은 건설 현장 관리자의 작업일지를 정리하는 한국어 비서입니다.\n'
        '아래 오늘의 소통 로그를 바탕으로 현장 보고서 형식의 요약을 작성하세요.\n'
        '반드시 다음 4개 제목을 모두 사용하고, 각 항목을 충분한 문장으로 구체적으로 작성하세요.\n'
        '1. 오늘의 핵심 작업\n'
        '2. 주요 지시 및 소통\n'
        '3. 위험 및 주의 사항\n'
        '4. 후속 조치\n\n'
        '주의사항:\n'
        '- 로그에 없는 내용은 추측하지 마세요.\n'
        '- 번역 기록, 작업자 호출/요청, 안전 알림, 수동 입력을 모두 참고해 작업 흐름을 복원하세요.\n'
        '- 긴 수동 입력은 그대로 길게 다시 쓰지 말고, 핵심 작업/지시/위험/후속조치만 압축해서 반영하세요.\n'
        '- 반복된 호출이나 동일한 알림은 묶어서 정리하세요.\n'
        '- 낙상 감지나 소음 경고가 있으면 반드시 위험 및 주의 사항에 반영하세요.\n'
        '- 각 항목은 단순 나열이 아니라 "로그 분석 결과 + 수동입력 핵심 요약"이 함께 드러나야 합니다.\n'
        '- 수동 입력에 긴 문단이 있으면 핵심 맥락과 후속 조치를 놓치지 말고 반영하세요.\n'
        '- 전체 분량은 지나치게 짧지 않게, 10~16문장 정도의 자연스러운 요약 보고서로 작성하세요.\n'
        '- 출력이 짧게 끊기지 않도록, 필요한 내용을 빠뜨리지 말고 보고서처럼 자연스럽게 이어서 작성하세요.\n'
        '- 불필요한 서론이나 사과 없이 바로 결과만 작성하세요.\n\n'
        f'[오늘 로그 메타데이터]\n{metadata_block}\n\n'
        f'[로그 분석 초안]\n{log_analysis_block}\n\n'
        f'[번역 기록]\n{translation_block}\n\n'
        f'[작업자 호출/요청 집계]\n{worker_block}\n\n'
        f'[안전 알림]\n{alert_block}\n\n'
        f'[수동 입력 핵심 요약]\n{manual_summary_block}\n\n'
        f'[수동 입력 원문/요약]\n{manual_block}\n\n'
        '[출력 형식 예시]\n'
        '## 현장 보고서\n'
        '### 1. 오늘의 핵심 작업\n'
        '- 오늘 실제로 진행된 작업과 점검 내용을 수동입력과 로그를 종합해 요약\n'
        '### 2. 주요 지시 및 소통\n'
        '- 작업자에게 전달한 주요 지시, 번역 기록, 호출/요청 패턴을 분석해 설명\n'
        '### 3. 위험 및 주의 사항\n'
        '- 소음 경고, 낙상 감지, 반복 호출 등 위험 신호를 요약 정리\n'
        '### 4. 후속 조치\n'
        '- 내일 확인할 점, 재점검 필요 사항, 후속 대응을 짧고 명확하게 정리'
    )



def _summary_has_required_sections(text: str) -> bool:
    required_titles = [
        '### 1. 오늘의 핵심 작업',
        '### 2. 주요 지시 및 소통',
        '### 3. 위험 및 주의 사항',
        '### 4. 후속 조치',
    ]
    return all(title in text for title in required_titles)


def _summary_looks_truncated(text: str) -> bool:
    stripped = (text or '').strip()
    if not stripped:
        return True
    if not _summary_has_required_sections(stripped):
        return True
    return stripped[-1] not in '.!?\n' and not stripped.endswith('다') and not stripped.endswith('요')


def _build_summary_retry_prompt(entries: list[dict]) -> str:
    return (
        _build_gemini_prompt_from_entries(entries)
        + '\n\n[재작성 지시]\n'
        + '이전 응답이 일부 잘리거나 항목이 누락되었습니다. 이번에는 반드시 아래 4개 제목을 모두 포함하고, 각 항목을 2~3문장의 완결된 문장으로 끝까지 작성하세요. 전체 분량은 지나치게 짧지 않게 10~16문장 정도로 유지하세요.\n'
        + '### 1. 오늘의 핵심 작업\n'
        + '### 2. 주요 지시 및 소통\n'
        + '### 3. 위험 및 주의 사항\n'
        + '### 4. 후속 조치\n'
        + '반드시 중간에 끊기지 않는 완결된 결과만 출력하고, 로그 분석 결과와 수동입력 핵심 요약을 함께 반영하세요.'
    )

def _build_today_summary_from_entries(entries: list[dict]) -> tuple[str, int, str]:
    if not entries:
        return _build_rule_based_summary_from_entries(entries)

    if is_gemini_configured():
        prompt = _build_gemini_prompt_from_entries(entries)
        result = analyze_text_with_meta(prompt, max_output_tokens=2400)
        summary_text = (result.get('text') or '').strip()
        finish_reason = str(result.get('finish_reason') or '').strip()
        model_name = (result.get('model_name') or _preferred_gemini_model_name())

        if summary_text and not str(result.get('error') or '').startswith('AI analysis failed'):
            if finish_reason == 'MAX_TOKENS' or _summary_looks_truncated(summary_text):
                retry_prompt = _build_summary_retry_prompt(entries)
                retry_result = analyze_text_with_meta(retry_prompt, max_output_tokens=2800)
                retry_text = (retry_result.get('text') or '').strip()
                if retry_text and not str(retry_result.get('error') or '').startswith('AI analysis failed'):
                    return retry_text, len(entries), str(retry_result.get('model_name') or model_name)
            return summary_text, len(entries), str(model_name)

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





