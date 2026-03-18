from datetime import datetime
from typing import List

from deep_translator import GoogleTranslator, MyMemoryTranslator
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from mongo_store import fetch_translation_history

router = APIRouter(prefix='/api', tags=['translations'])

MYMEMORY_LANGUAGE_MAP = {
    'ko': 'ko-KR',
    'en': 'en-US',
    'vi': 'vi-VN',
    'th': 'th-TH',
    'uz': 'uz-UZ',
    'mn': 'mn-MN',
    'zh-cn': 'zh-CN',
    'ja': 'ja-JP',
    'id': 'id-ID',
    'tl': 'tl-PH',
    'ne': 'ne-NP',
    'ru': 'ru-RU',
    'km': 'km-KH',
}


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    source_language: str = Field(default='ko', min_length=2, max_length=10)
    target_language: str = Field(..., min_length=2, max_length=10)

    @field_validator('text', mode='before')
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = (value or '').strip()
        if not cleaned:
            raise ValueError('text cannot be empty')
        return cleaned

    @field_validator('source_language', 'target_language', mode='before')
    @classmethod
    def normalize_lang(cls, value: str) -> str:
        cleaned = (value or '').strip().lower()
        if not cleaned:
            raise ValueError('language code cannot be empty')
        return cleaned


class TranslateResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str


class TranslationHistoryItem(BaseModel):
    id: str
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    created_at: datetime


@router.get('/translate/health')
def translation_health():
    return {'status': 'ok', 'service': 'translations'}


def translate_with_fallbacks(text: str, source_language: str, target_language: str) -> str:
    if source_language == target_language:
        return text

    source_mm = MYMEMORY_LANGUAGE_MAP.get(source_language, source_language)
    target_mm = MYMEMORY_LANGUAGE_MAP.get(target_language, target_language)

    attempts = [
        lambda: GoogleTranslator(source=source_language, target=target_language).translate(text),
        lambda: GoogleTranslator(source='auto', target=target_language).translate(text),
        lambda: MyMemoryTranslator(source=source_mm, target=target_mm).translate(text),
        lambda: MyMemoryTranslator(source='auto', target=target_mm).translate(text),
    ]

    last_error = None

    for translator in attempts:
        try:
            translated = translator()
            if translated and translated.strip():
                return translated.strip()
        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error

    raise RuntimeError('empty translation result')


@router.post('/translate', response_model=TranslateResponse)
def translate(payload: TranslateRequest):
    try:
        translated_text = translate_with_fallbacks(
            text=payload.text,
            source_language=payload.source_language,
            target_language=payload.target_language,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'translation provider error: {exc}') from exc


    return TranslateResponse(
        translated_text=translated_text,
        source_language=payload.source_language,
        target_language=payload.target_language,
    )


@router.get('/translations', response_model=List[TranslationHistoryItem])
def get_translations(limit: int = Query(default=20, ge=1, le=100)):
    rows = fetch_translation_history(limit=limit)
    return [TranslationHistoryItem(**row) for row in rows]
