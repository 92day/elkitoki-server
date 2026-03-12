import asyncio
import json
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
from models.models import Alert, SensorData, Zone

try:
    import serial
except ImportError:
    serial = None

load_dotenv()

router = APIRouter(prefix='/api/alerts', tags=['alerts'])
sensor_router = APIRouter(prefix='/api/sensors', tags=['sensors'])
connected_sensor_clients: list[WebSocket] = []


class AlertCreate(BaseModel):
    level: str
    message: str
    source: Optional[str] = 'Manual Input'
    zone_id: Optional[int] = None
    zone_name: Optional[str] = None


def resolve_zone_name(db: Session, zone_id: Optional[int], zone_name: Optional[str]) -> tuple[Optional[int], Optional[str]]:
    cleaned_name = zone_name.strip() if zone_name else None
    if zone_id is None:
        return None, cleaned_name or None

    zone = db.query(Zone).filter(Zone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=400, detail='Invalid zone_id. Check the zones table.')
    return zone_id, zone.name


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
    return alert


@router.patch('/{alert_id}/resolve')
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail='Alert not found.')

    alert.is_resolved = True
    db.commit()
    return {'message': 'Resolved'}


async def broadcast_sensor_update(data: dict):
    for client in connected_sensor_clients[:]:
        try:
            await client.send_json(data)
        except Exception:
            connected_sensor_clients.remove(client)


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
def get_latest_sensor_values(db: Session = Depends(get_db)):
    results = {}
    for sensor_type in ['temperature', 'humidity', 'dust', 'gas']:
        row = (
            db.query(SensorData)
            .filter(SensorData.sensor_type == sensor_type)
            .order_by(SensorData.recorded_at.desc())
            .first()
        )
        results[sensor_type] = {'value': row.value, 'unit': row.unit} if row else None
    return results


async def read_arduino_serial():
    if serial is None:
        print('[Arduino] pyserial is not installed. Skipping serial reader.')
        return

    port = os.getenv('ARDUINO_PORT', 'COM3')
    baud = int(os.getenv('ARDUINO_BAUDRATE', '9600'))

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

                    db = SessionLocal()
                    try:
                        zone_id = data.get('zone_id')
                        zone_name = None
                        if zone_id is not None:
                            zone = db.query(Zone).filter(Zone.id == zone_id).first()
                            zone_name = zone.name if zone else None

                        sensor_row = SensorData(
                            zone_id=zone_id,
                            sensor_type=data.get('type'),
                            value=data.get('value'),
                            unit=data.get('unit', ''),
                        )
                        db.add(sensor_row)

                        alert_message = check_sensor_threshold(data)
                        if alert_message:
                            db.add(
                                Alert(
                                    level='high',
                                    message=alert_message,
                                    source='Sensor Auto Detection',
                                    zone_id=zone_id,
                                    zone_name=zone_name,
                                )
                            )

                        db.commit()
                    finally:
                        db.close()

                    await broadcast_sensor_update({'event': 'sensor', 'data': data})
        except Exception as exc:
            print(f'[Arduino] Serial connection failed: {exc}. Retrying in 5 seconds.')
            await asyncio.sleep(5)


def check_sensor_threshold(data: dict):
    limits = {
        'temperature': (40, 'High temperature detected'),
        'humidity': (90, 'High humidity detected'),
        'dust': (150, 'High dust level detected'),
        'gas': (50, 'Gas leak detected'),
    }
    sensor_type = data.get('type')
    value = data.get('value', 0)

    if sensor_type in limits and isinstance(value, (int, float)):
        limit, message = limits[sensor_type]
        if value > limit:
            return f'[Sensor] {message}: {value}{data.get("unit", "")}'
    return None
