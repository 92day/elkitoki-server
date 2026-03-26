import os
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover
    MongoClient = None

load_dotenv()

_mongo_client = None
_mongo_db = None
ZONE_ID_BY_LABEL = {'A': 1, 'B': 2, 'C': 3}
WORKER_NAME_BY_KEY = {'A': '이레드', 'B': '김그린'}
DAILY_LOG_COLLECTIONS = ('translation_logs', 'worker_call_logs', 'manual_logs', 'alert_logs')
REPORT_LOG_COLLECTIONS = ('translation_logs', 'worker_call_logs', 'manual_logs')


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


def insert_document(collection_name: str, document: dict[str, Any]):
    db = get_mongo_db()
    if db is None:
        return None
    return db[collection_name].insert_one(document)


def delete_document(collection_name: str, field_name: str, field_value: Any) -> None:
    db = get_mongo_db()
    if db is None:
        return
    db[collection_name].delete_one({field_name: field_value})


def delete_document_by_id(collection_name: str, mongo_id: str) -> None:
    db = get_mongo_db()
    if db is None:
        return
    db[collection_name].delete_one({'_id': ObjectId(mongo_id)})


def _resolve_report_collection_by_values(*, entry_type: str, text_content: str) -> str:
    text = (text_content or '').strip()
    normalized_entry_type = (entry_type or '').strip().lower()

    if text.startswith('[작업자 호출]') or text.startswith('[작업자 요청]'):
        return 'worker_call_logs'
    if normalized_entry_type == 'translation':
        return 'translation_logs'
    return 'manual_logs'


def _resolve_report_collection(report) -> str:
    return _resolve_report_collection_by_values(
        entry_type=getattr(report, 'entry_type', '') or '',
        text_content=getattr(report, 'text_content', '') or '',
    )


def _normalize_report_document(collection_name: str, document: dict[str, Any]) -> dict[str, Any]:
    mongo_id = str(document.get('_id'))
    if collection_name == 'translation_logs':
        text_content = document.get('source_text') or document.get('text_content') or ''
        translated_text = document.get('translated_text') or ''
        entry_type = 'translation'
    else:
        text_content = document.get('text_content') or document.get('message') or ''
        translated_text = document.get('translated_text') or None
        entry_type = document.get('entry_type') or 'manual'

    author_name = (
        document.get('author_name')
        or document.get('source')
        or ('번역기' if collection_name == 'translation_logs' else '시스템')
    )

    return {
        'id': f'{collection_name}:{mongo_id}',
        'mongo_id': mongo_id,
        'collection_name': collection_name,
        'date': document.get('date'),
        'entry_type': entry_type,
        'text_content': text_content,
        'translated_text': translated_text,
        'source_language': document.get('source_language') or 'ko',
        'target_language': document.get('target_language') or 'ko',
        'author_name': author_name,
        'created_at': document.get('created_at') or _now_iso(),
    }


def insert_report_entry(
    *,
    entry_type: str,
    text_content: str,
    translated_text: str | None = None,
    source_language: str = 'ko',
    target_language: str = 'ko',
    author_name: str = 'Site Manager',
    created_at: str | None = None,
    date_text: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    timestamp = created_at or _now_iso()
    today_text = date_text or timestamp[:10]
    collection_name = _resolve_report_collection_by_values(entry_type=entry_type, text_content=text_content)

    document: dict[str, Any] = {
        'date': today_text,
        'entry_type': entry_type,
        'text_content': text_content,
        'translated_text': translated_text or None,
        'source_language': source_language,
        'target_language': target_language,
        'author_name': author_name,
        'created_at': timestamp,
    }

    if collection_name == 'translation_logs':
        document['source_text'] = text_content
    if source:
        document['source'] = source

    result = insert_document(collection_name, document)
    if result is not None:
        document['_id'] = result.inserted_id
    else:
        document['_id'] = ObjectId()

    return _normalize_report_document(collection_name, document)


def insert_worker_request_log(
    *,
    worker: str | None,
    source: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    worker_label = WORKER_NAME_BY_KEY.get(str(worker), '작업자')
    return insert_report_entry(
        entry_type='manual',
        text_content=f'[작업자 요청] {worker_label} 요청',
        source_language='ko',
        target_language='ko',
        author_name='현장 버튼',
        source=source,
        created_at=created_at,
    )


def sync_report_log(report) -> None:
    insert_document(
        _resolve_report_collection(report),
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
    delete_document(_resolve_report_collection(report), 'mysql_report_id', getattr(report, 'id', None))


def insert_translation_request_log(
    *,
    source_text: str,
    translated_text: str,
    source_language: str,
    target_language: str,
) -> None:
    insert_document(
        'translation_logs',
        {
            'source_text': source_text,
            'translated_text': translated_text,
            'source_language': source_language,
            'target_language': target_language,
            'log_origin': 'translation_request',
            'created_at': _now_iso(),
        },
    )


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
            'status': getattr(alert, 'status', None),
            'is_resolved': getattr(alert, 'is_resolved', None),
            'handled_at': getattr(alert, 'handled_at', None).isoformat() if getattr(alert, 'handled_at', None) else None,
            'event_type': event_type,
            'created_at': getattr(alert, 'created_at', None).isoformat() if getattr(alert, 'created_at', None) else _now_iso(),
        },
    )


def delete_alert_logs(mysql_alert_id: Any) -> None:
    db = get_mongo_db()
    if db is None:
        return
    db['alert_logs'].delete_many({'mysql_alert_id': mysql_alert_id})


def sync_sensor_status_log(payload: dict[str, Any]) -> None:
    timestamp = payload.get('timestamp') or _now_iso()
    zones = []
    for zone_label in ('A', 'B', 'C'):
        if f'sound{zone_label}' in payload:
            zones.append(zone_label)

    insert_document(
        'sensor_status_logs',
        {
            'kind': payload.get('kind') or 'status',
            'device': payload.get('device'),
            'timestamp': timestamp,
            'zones': zones,
            'payload': payload,
            'created_at': timestamp,
        },
    )


def sync_sensor_event_log(payload: dict[str, Any]) -> None:
    timestamp = payload.get('timestamp') or _now_iso()
    zone = (payload.get('zone') or '').strip().upper() or None
    zone_id = payload.get('zone_id')
    if zone_id is None and zone in ZONE_ID_BY_LABEL:
        zone_id = ZONE_ID_BY_LABEL[zone]

    insert_document(
        'sensor_event_logs',
        {
            'kind': payload.get('kind') or 'event',
            'device': payload.get('device'),
            'event_type': payload.get('eventType'),
            'timestamp': timestamp,
            'zone': zone,
            'zone_id': zone_id,
            'payload': payload,
            'created_at': timestamp,
        },
    )


def _find_today_documents(collection_name: str, today_text: str) -> list[dict[str, Any]]:
    db = get_mongo_db()
    if db is None:
        return []

    try:
        today_date = datetime.strptime(today_text, '%Y-%m-%d').date()
        previous_date_text = (today_date - timedelta(days=1)).isoformat()
    except ValueError:
        previous_date_text = today_text

    documents = list(
        db[collection_name]
        .find(
            {
                '$or': [
                    {'date': today_text},
                    {'created_at': {'$regex': f'^{today_text}'}},
                    {'created_at': {'$regex': f'^{previous_date_text}'}},
                ]
            }
        )
        .sort('created_at', -1)
    )

    filtered: list[dict[str, Any]] = []
    for document in documents:
        if str(document.get('date') or '').strip() == today_text:
            filtered.append(document)
            continue

        created_at = str(document.get('created_at') or '').strip()
        if not created_at:
            continue

        try:
            parsed = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            local_date = parsed.astimezone().date().isoformat()
            if local_date == today_text:
                filtered.append(document)
        except ValueError:
            if created_at.startswith(today_text):
                filtered.append(document)

    return filtered


def fetch_translation_history(limit: int = 20) -> list[dict[str, Any]]:
    db = get_mongo_db()
    if db is None:
        return []

    cursor = (
        db['translation_logs']
        .find(
            {
                '$or': [
                    {'source_text': {'$exists': True}},
                    {'text_content': {'$exists': True}},
                ]
            }
        )
        .sort('created_at', -1)
        .limit(max(1, min(limit, 100)))
    )

    rows: list[dict[str, Any]] = []
    for document in cursor:
        rows.append(
            {
                'id': str(document.get('_id')),
                'source_text': document.get('source_text') or document.get('text_content') or '',
                'translated_text': document.get('translated_text') or '',
                'source_language': document.get('source_language') or 'ko',
                'target_language': document.get('target_language') or 'ko',
                'created_at': document.get('created_at') or _now_iso(),
            }
        )
    return rows


def _parse_created_at(value: Any) -> datetime:
    created_at = str(value or '').strip()
    if not created_at:
        return datetime.min.replace(tzinfo=timezone.utc)

    try:
        parsed = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

def fetch_today_daily_log_entries(today_text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    mappings = [
        ('translation_logs', 'translation'),
        ('worker_call_logs', 'worker_call'),
        ('manual_logs', 'manual'),
        ('alert_logs', 'alert'),
    ]

    for collection_name, log_type in mappings:
        for document in _find_today_documents(collection_name, today_text):
            mongo_id = str(document.get('_id'))
            text_content = (
                document.get('text_content')
                or document.get('source_text')
                or document.get('message')
                or ''
            )
            author_name = (
                document.get('author_name')
                or document.get('source')
                or ('번역기' if log_type == 'translation' else '시스템')
            )
            mysql_report_id = document.get('mysql_report_id')

            created_at = document.get('created_at') or _now_iso()
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
                    'created_at': created_at,
                    'mysql_report_id': mysql_report_id,
                    'deletable': True,
                    'level': document.get('level'),
                    'zone_id': document.get('zone_id'),
                    'zone_name': document.get('zone_name'),
                    'event_type': document.get('event_type'),
                }
            )

    entries.sort(key=lambda item: _parse_created_at(item.get('created_at')), reverse=True)
    return entries


def fetch_today_report_entries(today_text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for collection_name in REPORT_LOG_COLLECTIONS:
        for document in _find_today_documents(collection_name, today_text):
            entries.append(_normalize_report_document(collection_name, document))

    entries.sort(key=lambda item: _parse_created_at(item.get('created_at')), reverse=True)
    return entries


def clear_today_daily_log_entries(today_text: str) -> None:
    db = get_mongo_db()
    if db is None:
        return

    for collection_name in DAILY_LOG_COLLECTIONS:
        for document in _find_today_documents(collection_name, today_text):
            db[collection_name].delete_one({'_id': document.get('_id')})


def _build_sensor_log_filter(
    *,
    date_text: str | None = None,
    device: str | None = None,
    event_type: str | None = None,
    zone: str | None = None,
    is_event: bool = False,
) -> dict[str, Any]:
    query: dict[str, Any] = {}

    if date_text:
        query['created_at'] = {'$regex': f'^{date_text}'}
    if device:
        query['device'] = device
    if is_event and event_type:
        query['event_type'] = event_type
    if zone:
        normalized_zone = zone.strip().upper()
        zone_id = ZONE_ID_BY_LABEL.get(normalized_zone)
        if is_event:
            zone_values = [normalized_zone]
            if normalized_zone and not normalized_zone.endswith('구역'):
                zone_values.append(f'{normalized_zone}구역')
            or_clauses: list[dict[str, Any]] = [
                {'zone': {'$in': zone_values}},
                {'payload.zone': {'$in': zone_values}},
            ]
            if zone_id is not None:
                or_clauses.append({'zone_id': zone_id})
                or_clauses.append({'payload.zone_id': zone_id})
            query['$or'] = or_clauses
        else:
            query['zones'] = normalized_zone

    return query


def _normalize_sensor_document(document: dict[str, Any], *, is_event: bool) -> dict[str, Any]:
    mongo_id = str(document.get('_id'))
    normalized = {
        'id': mongo_id,
        'mongo_id': mongo_id,
        'kind': document.get('kind'),
        'device': document.get('device'),
        'timestamp': document.get('timestamp'),
        'created_at': document.get('created_at') or _now_iso(),
        'payload': document.get('payload') or {},
    }
    if is_event:
        normalized['event_type'] = document.get('event_type')
        normalized['zone'] = document.get('zone')
        normalized['zone_id'] = document.get('zone_id')
    else:
        normalized['zones'] = document.get('zones') or []
    return normalized


def fetch_sensor_status_logs(
    *,
    date_text: str | None = None,
    device: str | None = None,
    zone: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    db = get_mongo_db()
    if db is None:
        return []

    cursor = (
        db['sensor_status_logs']
        .find(_build_sensor_log_filter(date_text=date_text, device=device, zone=zone, is_event=False))
        .sort('created_at', -1)
        .limit(max(1, min(limit, 500)))
    )
    return [_normalize_sensor_document(document, is_event=False) for document in cursor]


def fetch_sensor_event_logs(
    *,
    date_text: str | None = None,
    device: str | None = None,
    event_type: str | None = None,
    zone: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    db = get_mongo_db()
    if db is None:
        return []

    cursor = (
        db['sensor_event_logs']
        .find(_build_sensor_log_filter(date_text=date_text, device=device, event_type=event_type, zone=zone, is_event=True))
        .sort('created_at', -1)
        .limit(max(1, min(limit, 500)))
    )
    return [_normalize_sensor_document(document, is_event=True) for document in cursor]


