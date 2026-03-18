import asyncio
import json
import os
from datetime import date, datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Body, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
from models.models import Alert, Report, SensorData, Zone
from mongo_store import sync_alert_log, sync_report_log, sync_sensor_event_log, sync_sensor_status_log

try:
    import serial
except ImportError:
    serial = None

load_dotenv()

router = APIRouter(prefix='/api/alerts', tags=['alerts'])
sensor_router = APIRouter(prefix='/api/sensors', tags=['sensors'])
device_router = APIRouter(prefix='/api/device', tags=['device'])
connected_sensor_clients: list[WebSocket] = []
latest_sensor_cache: dict[str, dict[str, Any]] = {}
device_command_queue: list[dict[str, Any]] = []
next_command_id = 1

STATUS_EXCLUDE_KEYS = {'kind', 'device', 'timestamp'}
ZONE_ID_BY_LABEL = {'A': 1, 'B': 2, 'C': 3}
ZONE_SOUND_FIELDS = {1: 'soundA', 2: 'soundB', 3: 'soundC'}
last_valid_zone_noise: dict[int, dict[str, Any]] = {}
NOISE_STALE_SECONDS = 4


class AlertCreate(BaseModel):
    level: str
    message: str
    source: Optional[str] = 'Manual Input'
    zone_id: Optional[int] = None
    zone_name: Optional[str] = None


class DeviceCommandCreate(BaseModel):
    device: str = 'uno-main'
    cmd: str
    worker: Optional[str] = None
    color: Optional[str] = None
    state: Optional[str] = None
    payload: dict[str, Any] = {}


class DeviceCommandAck(BaseModel):
    bridge_id: Optional[str] = None


def resolve_zone_name(db: Session, zone_id: Optional[int], zone_name: Optional[str]) -> tuple[Optional[int], Optional[str]]:
    cleaned_name = zone_name.strip() if zone_name else None
    if zone_id is None:
        return None, cleaned_name or None

    zone = db.query(Zone).filter(Zone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=400, detail='Invalid zone_id. Check the zones table.')
    return zone_id, zone.name


def map_zone_name_to_id(zone_name: Optional[str]) -> Optional[int]:
    if not zone_name:
        return None

    normalized = zone_name.strip().upper()
    if normalized in ZONE_ID_BY_LABEL:
        return ZONE_ID_BY_LABEL[normalized]
    if normalized.startswith('ZONE '):
        return ZONE_ID_BY_LABEL.get(normalized.replace('ZONE ', '', 1))
    return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def coerce_noise_score(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float)):
        return None

    numeric = float(value)
    if numeric <= 0:
        return 35
    if numeric <= 120:
        return int(round(numeric))

    scaled = 35 + (min(numeric, 1023.0) / 1023.0) * 60
    return int(round(scaled))


def classify_noise_status(score: Optional[int]) -> str:
    if score is None:
        return 'safe'
    if score >= 70:
        return 'danger'
    if score >= 40:
        return 'caution'
    return 'safe'


def format_peak_time(timestamp: Optional[str]) -> str:
    if not timestamp:
        return '--:--'
    try:
        parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return parsed.astimezone().strftime('%H:%M')
    except ValueError:
        return '--:--'


def build_zone_noise_payload() -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}

    for zone_id, field_name in ZONE_SOUND_FIELDS.items():
        cached = latest_sensor_cache.get(field_name) or {}
        score = coerce_noise_score(cached.get('value'))
        updated_at = cached.get('updatedAt')

        if score is not None:
            last_valid_zone_noise[zone_id] = {
                'score': score,
                'peak': format_peak_time(updated_at),
                'status': classify_noise_status(score),
                'updatedAt': updated_at,
            }

        fallback = last_valid_zone_noise.get(zone_id)
        if fallback and fallback.get('updatedAt'):
            try:
                fallback_dt = datetime.fromisoformat(str(fallback['updatedAt']).replace('Z', '+00:00'))
                age_seconds = (datetime.now(timezone.utc) - fallback_dt.astimezone(timezone.utc)).total_seconds()
                if age_seconds > NOISE_STALE_SECONDS:
                    fallback = None
            except ValueError:
                fallback = None

        payload[str(zone_id)] = fallback or {
            'score': 1,
            'peak': '--:--',
            'status': 'safe',
            'updatedAt': None,
        }

    return payload


def add_sensor_row(
    db: Session,
    *,
    sensor_type: str,
    value: Any,
    unit: str = '',
    zone_id: Optional[int] = None,
) -> None:
    if isinstance(value, bool):
        numeric_value = 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        numeric_value = float(value)
    else:
        return

    db.add(SensorData(zone_id=zone_id, sensor_type=sensor_type, value=numeric_value, unit=unit))


def update_latest_cache_from_payload(payload: dict[str, Any]) -> None:
    kind = payload.get('kind')
    timestamp = payload.get('timestamp') or now_iso()
    device = payload.get('device')

    if kind == 'status':
        for key, value in payload.items():
            if key in STATUS_EXCLUDE_KEYS:
                continue
            if isinstance(value, (int, float, bool, str)):
                latest_sensor_cache[key] = {
                    'value': value,
                    'updatedAt': timestamp,
                    'device': device,
                }
        return

    if kind == 'event':
        event_type = payload.get('eventType')
        if not event_type:
            return

        latest_sensor_cache[f'event:{event_type}'] = {
            'value': payload,
            'updatedAt': timestamp,
            'device': device,
        }

        if event_type == 'worker_call_button':
            worker = payload.get('worker')
            if worker in {'A', 'B'}:
                latest_sensor_cache[f'callLast{worker}'] = {
                    'value': payload.get('source') or 'manual',
                    'updatedAt': timestamp,
                    'device': device,
                }

        if event_type == 'fall_detected':
            latest_sensor_cache['fallDetected'] = {
                'value': bool(payload.get('active')),
                'updatedAt': timestamp,
                'device': device,
            }


def persist_sensor_rows(db: Session, payload: dict[str, Any]) -> None:
    kind = payload.get('kind')
    zone_id = payload.get('zone_id')

    if kind == 'status':
        status_mappings = [
            ('heartRawA', 'heart_raw_a', '', None),
            ('heartRawB', 'heart_raw_b', '', None),
            ('fingerA', 'finger_detected_a', '', None),
            ('fingerB', 'finger_detected_b', '', None),
            ('soundA', 'sound_level_a', 'raw', 1),
            ('soundB', 'sound_level_b', 'raw', 2),
            ('soundC', 'sound_level_c', 'raw', 3),
            ('soundAlertA', 'sound_alert_a', '', 1),
            ('soundAlertB', 'sound_alert_b', '', 2),
            ('soundAlertC', 'sound_alert_c', '', 3),
            ('tiltAlert', 'fall_state', '', None),
            ('nanoConnected', 'nano_connected', '', None),
            ('pitch', 'tilt_pitch', 'deg', None),
            ('roll', 'tilt_roll', 'deg', None),
            ('buttonPressedA', 'button_pressed_a', '', None),
            ('buttonPressedB', 'button_pressed_b', '', None),
            ('callActiveA', 'call_active_a', '', None),
            ('callActiveB', 'call_active_b', '', None),
        ]
        for field_name, sensor_type, unit, mapped_zone_id in status_mappings:
            add_sensor_row(
                db,
                sensor_type=sensor_type,
                value=payload.get(field_name),
                unit=unit,
                zone_id=mapped_zone_id if mapped_zone_id is not None else zone_id,
            )
        return

    if kind == 'event':
        event_type = payload.get('eventType')
        active = bool(payload.get('active'))

        if event_type == 'fall_detected':
            add_sensor_row(db, sensor_type='fall_detected', value=active, zone_id=zone_id)
        elif event_type == 'heart_abnormal':
            worker = payload.get('worker')
            sensor_type = f'heart_alert_{str(worker).lower()}' if worker in {'A', 'B'} else 'heart_alert'
            add_sensor_row(db, sensor_type=sensor_type, value=payload.get('value'), unit='raw', zone_id=zone_id)
        elif event_type == 'worker_call_button':
            worker = payload.get('worker')
            sensor_type = f'worker_call_{str(worker).lower()}' if worker in {'A', 'B'} else 'worker_call'
            add_sensor_row(db, sensor_type=sensor_type, value=active, zone_id=zone_id)
        elif event_type == 'noise_abnormal':
            event_zone_id = zone_id or map_zone_name_to_id(payload.get('zone'))
            add_sensor_row(db, sensor_type='noise_event', value=payload.get('value'), unit='raw', zone_id=event_zone_id)


def build_alert_from_payload(payload: dict[str, Any], db: Session) -> Optional[Alert]:
    if payload.get('kind') != 'event' or not payload.get('active'):
        return None

    event_type = payload.get('eventType')
    zone_name = payload.get('zone')
    zone_id = payload.get('zone_id') or map_zone_name_to_id(zone_name)
    resolved_zone_name = None

    if zone_id is not None:
        zone = db.query(Zone).filter(Zone.id == zone_id).first()
        resolved_zone_name = zone.name if zone else None

    if event_type == 'noise_abnormal':
        zone_label = zone_name or resolved_zone_name or 'A'
        value = payload.get('value')
        return Alert(
            level='high',
            message=f'{zone_label} 구역 소음지수 경고: {value}',
            source='Sound Sensor',
            zone_id=zone_id,
            zone_name=resolved_zone_name or zone_label,
        )

    if event_type == 'fall_detected':
        return Alert(
            level='high',
            message='작업자 낙상 감지!',
            source='Nano Tilt',
            zone_id=zone_id,
            zone_name=resolved_zone_name,
        )

    if event_type == 'worker_call_button':
        source = str(payload.get('source', 'manual_button'))
        worker = payload.get('worker')
        if source.startswith('manual_button'):
            return None

        worker_label = f'작업자 {worker}' if worker in {'A', 'B'} else '작업자'
        return Alert(
            level='mid',
            message=f'{worker_label} 호출 발생: 대시보드 호출',
            source='Call Input',
            zone_id=zone_id,
            zone_name=resolved_zone_name,
        )

    return None


def build_report_from_payload(payload: dict[str, Any]) -> Optional[Report]:
    if payload.get('kind') != 'event' or not payload.get('active'):
        return None

    if payload.get('eventType') != 'worker_call_button':
        return None

    source = str(payload.get('source', ''))
    if not source.startswith('manual_button'):
        return None

    worker = payload.get('worker')
    worker_label = f'작업자 {worker}' if worker in {'A', 'B'} else '작업자'
    return Report(
        date=str(date.today()),
        entry_type='manual',
        text_content=f'[작업자 요청] {worker_label} 요청',
        translated_text=None,
        source_language='ko',
        target_language='ko',
        author_name='현장 버튼',
    )


async def broadcast_sensor_update(data: dict[str, Any]) -> None:
    for client in connected_sensor_clients[:]:
        try:
            await client.send_json(data)
        except Exception:
            connected_sensor_clients.remove(client)


async def process_sensor_payload(payload: dict[str, Any]) -> None:
    payload.setdefault('timestamp', now_iso())
    update_latest_cache_from_payload(payload)

    if payload.get('kind') == 'status':
        sync_sensor_status_log(payload)
    elif payload.get('kind') == 'event':
        sync_sensor_event_log(payload)

    db = SessionLocal()
    try:
        persist_sensor_rows(db, payload)
        alert = build_alert_from_payload(payload, db)
        if alert:
            db.add(alert)
        report = build_report_from_payload(payload)
        if report:
            db.add(report)
        db.commit()
        if alert:
            db.refresh(alert)
            sync_alert_log(alert, payload.get('eventType'))
        if report:
            db.refresh(report)
            sync_report_log(report)
    finally:
        db.close()

    await broadcast_sensor_update({'event': 'sensor', 'data': payload})


@router.get('/')
def get_alerts(db: Session = Depends(get_db)):
    return db.query(Alert).filter(Alert.is_resolved.is_(False)).order_by(Alert.created_at.desc()).all()


@router.post('/')
def create_alert(data: AlertCreate, db: Session = Depends(get_db)):
    zone_id, zone_name = resolve_zone_name(db, data.zone_id, data.zone_name)
    alert = Alert(
        level=data.level,
        message=data.message,
        source=data.source,
        zone_id=zone_id,
        zone_name=zone_name,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    sync_alert_log(alert, 'manual_alert')
    return alert


@router.patch('/{alert_id}/resolve')
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail='Alert not found.')

    alert.is_resolved = True
    db.commit()
    return {'message': 'Resolved'}


@sensor_router.websocket('/ws')
async def sensor_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_sensor_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connected_sensor_clients:
            connected_sensor_clients.remove(websocket)


@sensor_router.get('/latest')
def get_latest_sensor_values():
    return {
        **latest_sensor_cache,
        'zoneNoiseById': build_zone_noise_payload(),
    }


@sensor_router.post('/events')
async def receive_sensor_event(payload: dict[str, Any] = Body(...)):
    await process_sensor_payload(payload)
    return {'ok': True}


@device_router.post('/commands')
def create_device_command(payload: DeviceCommandCreate):
    global next_command_id

    command = payload.model_dump()
    command['id'] = next_command_id
    command['status'] = 'pending'
    command['createdAt'] = now_iso()
    next_command_id += 1
    device_command_queue.append(command)
    return command


@device_router.get('/commands/pending')
def get_pending_device_commands(device: str):
    return [command for command in device_command_queue if command.get('device') == device and command.get('status') == 'pending']


@device_router.post('/commands/{command_id}/ack')
def ack_device_command(command_id: int, payload: DeviceCommandAck):
    for command in device_command_queue:
        if command['id'] == command_id:
            command['status'] = 'sent'
            command['ackedAt'] = now_iso()
            command['bridgeId'] = payload.bridge_id
            return {'ok': True}
    raise HTTPException(status_code=404, detail='Device command not found.')


async def read_arduino_serial():
    if serial is None:
        print('[Arduino] pyserial is not installed. Skipping serial reader.')
        return

    port = os.getenv('ARDUINO_PORT', 'COM3')
    baud = int(os.getenv('ARDUINO_BAUDRATE', '115200'))

    while True:
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                print(f'[Arduino] Connected to {port}')
                while True:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    await process_sensor_payload(data)
        except Exception as exc:
            print(f'[Arduino] Serial connection failed: {exc}. Retrying in 5 seconds.')
            await asyncio.sleep(5)





