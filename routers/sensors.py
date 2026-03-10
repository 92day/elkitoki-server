from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from database import get_db
from models.models import SensorData, Alert
import serial, asyncio, json, os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter(prefix="/api/sensors", tags=["sensors"])
connected_clients: list[WebSocket] = []


async def broadcast(data: dict):
    for client in connected_clients[:]:
        try:
            await client.send_json(data)
        except:
            connected_clients.remove(client)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@router.get("/latest")
def get_latest(db: Session = Depends(get_db)):
    results = {}
    for t in ["temperature", "humidity", "dust", "gas"]:
        row = db.query(SensorData).filter(SensorData.sensor_type == t).order_by(SensorData.recorded_at.desc()).first()
        results[t] = {"value": row.value, "unit": row.unit} if row else None
    return results


async def read_arduino_serial():
    port = os.getenv("ARDUINO_PORT", "COM3")
    baud = int(os.getenv("ARDUINO_BAUDRATE", 9600))
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            print(f"[Arduino] {port} 연결 성공")
            while True:
                line = ser.readline().decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    db = next(get_db())
                    db.add(SensorData(
                        zone_id=data.get("zone_id"),
                        sensor_type=data.get("type"),
                        value=data.get("value"),
                        unit=data.get("unit", ""),
                    ))
                    msg = _check_threshold(data)
                    if msg:
                        db.add(Alert(level="high", message=msg, source="센서 자동감지"))
                    db.commit()
                    await broadcast({"event": "sensor", "data": data})
                except json.JSONDecodeError:
                    pass
        except serial.SerialException as e:
            print(f"[Arduino] 연결 실패: {e} — 5초 후 재시도")
            await asyncio.sleep(5)


def _check_threshold(data: dict):
    limits = {"temperature": (40, "온도 위험"), "humidity": (90, "습도 위험"), "dust": (150, "미세먼지 위험"), "gas": (50, "가스 누출 감지")}
    t, v = data.get("type"), data.get("value", 0)
    if t in limits:
        limit, msg = limits[t]
        if v > limit:
            return f"[자동감지] {msg}: {v}{data.get('unit','')}"
    return None
