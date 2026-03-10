import asyncio
import json
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
from models.models import Alert, SensorData

try:
    import serial
except ImportError:
    serial = None

load_dotenv()
router = APIRouter(prefix='/api/sensors', tags=['sensors'])
connected_clients: list[WebSocket] = []


async def broadcast(data: dict):
    for client in connected_clients[:]:
        try:
            await client.send_json(data)
        except Exception:
            connected_clients.remove(client)


@router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@router.get('/latest')
def get_latest(db: Session = Depends(get_db)):
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
                        sensor_row = SensorData(
                            zone_id=data.get('zone_id'),
                            sensor_type=data.get('type'),
                            value=data.get('value'),
                            unit=data.get('unit', ''),
                        )
                        db.add(sensor_row)

                        alert_message = _check_threshold(data)
                        if alert_message:
                            db.add(Alert(level='high', message=alert_message, source='Sensor Auto Detection'))

                        db.commit()
                    finally:
                        db.close()

                    await broadcast({'event': 'sensor', 'data': data})
        except Exception as exc:
            print(f'[Arduino] Serial connection failed: {exc}. Retrying in 5 seconds.')
            await asyncio.sleep(5)


def _check_threshold(data: dict):
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
