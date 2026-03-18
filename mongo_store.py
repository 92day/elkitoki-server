import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover
    MongoClient = None

load_dotenv()

_mongo_client = None
_mongo_db = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_mongo_enabled() -> bool:
    return bool(MongoClient and os.getenv('MONGODB_URI') and os.getenv('MONGODB_DB_NAME'))


def get_mongo_db():
    global _mongo_client, _mongo_db

    if _mongo_db is not None:
        return _mongo_db

    if not is_mongo_enabled():
        return None

    _mongo_client = MongoClient(os.getenv('MONGODB_URI'), serverSelectionTimeoutMS=2000)
    _mongo_db = _mongo_client[os.getenv('MONGODB_DB_NAME')]
    return _mongo_db


def insert_document(collection_name: str, document: dict[str, Any]) -> None:
    db = get_mongo_db()
    if db is None:
        return
    db[collection_name].insert_one(document)


def delete_document(collection_name: str, field_name: str, field_value: Any) -> None:
    db = get_mongo_db()
    if db is None:
        return
    db[collection_name].delete_one({field_name: field_value})


def _resolve_report_collection(report) -> str:
    text = (getattr(report, 'text_content', '') or '').strip()
    entry_type = (getattr(report, 'entry_type', '') or '').strip()

    if text.startswith('[작업자 호출]') or text.startswith('[작업자 요청]'):
        return 'worker_call_logs'
    if entry_type == 'translation':
        return 'translation_logs'
    return 'manual_logs'


def sync_report_log(report) -> None:
    collection_name = _resolve_report_collection(report)
    insert_document(
        collection_name,
        {
            'mysql_report_id': getattr(report, 'id', None),
            'date': getattr(report, 'date', None),
            'entry_type': getattr(report, 'entry_type', None),
            'text_content': getattr(report, 'text_content', None),
            'translated_text': getattr(report, 'translated_text', None),
            'source_language': getattr(report, 'source_language', None),
            'target_language': getattr(report, 'target_language', None),
            'author_name': getattr(report, 'author_name', None),
            'created_at': getattr(report, 'created_at', None).isoformat() if getattr(report, 'created_at', None) else _now_iso(),
        },
    )


def delete_report_log(report) -> None:
    collection_name = _resolve_report_collection(report)
    delete_document(collection_name, 'mysql_report_id', getattr(report, 'id', None))


def sync_alert_log(alert, event_type: str | None = None) -> None:
    insert_document(
        'alert_logs',
        {
            'mysql_alert_id': getattr(alert, 'id', None),
            'level': getattr(alert, 'level', None),
            'message': getattr(alert, 'message', None),
            'source': getattr(alert, 'source', None),
            'zone_id': getattr(alert, 'zone_id', None),
            'zone_name': getattr(alert, 'zone_name', None),
            'event_type': event_type,
            'created_at': getattr(alert, 'created_at', None).isoformat() if getattr(alert, 'created_at', None) else _now_iso(),
        },
    )


def _find_today_documents(collection_name: str, today_text: str) -> list[dict[str, Any]]:
    db = get_mongo_db()
    if db is None:
        return []

    return list(
        db[collection_name]
        .find({'created_at': {'$regex': f'^{today_text}'}})
        .sort('created_at', 1)
    )


def fetch_today_daily_log_entries(today_text: str) -> list[dict[str, Any]]:
    mappings = [
        ('translation_logs', 'translation'),
        ('worker_call_logs', 'worker_call'),
        ('manual_logs', 'manual'),
        ('alert_logs', 'alert'),
    ]

    entries: list[dict[str, Any]] = []
    for collection_name, log_type in mappings:
        for document in _find_today_documents(collection_name, today_text):
            mongo_id = str(document.get('_id'))
            text_content = document.get('text_content') or document.get('message') or ''
            author_name = document.get('author_name') or document.get('source') or '시스템'
            mysql_report_id = document.get('mysql_report_id')

            entries.append(
                {
                    'id': f'{collection_name}:{mongo_id}',
                    'mongo_id': mongo_id,
                    'collection_name': collection_name,
                    'log_type': log_type,
                    'entry_type': 'translation' if log_type == 'translation' else 'manual',
                    'text_content': text_content,
                    'translated_text': document.get('translated_text'),
                    'author_name': author_name,
                    'source_language': document.get('source_language') or 'ko',
                    'target_language': document.get('target_language') or 'ko',
                    'created_at': document.get('created_at') or _now_iso(),
                    'mysql_report_id': mysql_report_id,
                    'deletable': mysql_report_id is not None,
                    'level': document.get('level'),
                    'zone_id': document.get('zone_id'),
                    'zone_name': document.get('zone_name'),
                }
            )

    entries.sort(key=lambda item: item.get('created_at') or '')
    return entries
