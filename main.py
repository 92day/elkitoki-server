from contextlib import asynccontextmanager
import asyncio
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text

from database import SessionLocal, engine
from models.models import Base, Zone
from routers import alerts, photos, report, sensors, weather, workers


def seed_default_zones():
    db = SessionLocal()
    try:
        default_zones = [
            {"id": 1, "name": "A구역", "description": "지하2층", "task": "철근 작업", "risk_level": "safe"},
            {"id": 2, "name": "B구역", "description": "지하1층", "task": "콘크리트", "risk_level": "safe"},
            {"id": 3, "name": "C구역", "description": "1~3층", "task": "고소 작업", "risk_level": "caution"},
            {"id": 4, "name": "D구역", "description": "4~6층", "task": "골조 공사", "risk_level": "safe"},
            {"id": 5, "name": "E구역", "description": "옥상", "task": "옥상 작업", "risk_level": "danger"},
            {"id": 6, "name": "F구역", "description": "외부", "task": "외벽 마감", "risk_level": "safe"},
        ]

        existing_ids = {
            row[0] for row in db.query(Zone.id).filter(Zone.id.in_([1, 2, 3, 4, 5, 6])).all()
        }
        for zone in default_zones:
            if zone["id"] not in existing_ids:
                db.add(Zone(**zone))
        db.commit()
    finally:
        db.close()


def patch_legacy_schema():
    with engine.begin() as conn:
        inspector = inspect(conn)
        table_names = set(inspector.get_table_names())

        if "photos" in table_names:
            photo_columns = {c["name"] for c in inspector.get_columns("photos")}
            if "risk_detected" not in photo_columns:
                conn.execute(
                    text("ALTER TABLE photos ADD COLUMN risk_detected BOOLEAN DEFAULT 0")
                )

        if "reports" in table_names:
            report_columns = {c["name"] for c in inspector.get_columns("reports")}
            if "ai_analysis" not in report_columns:
                conn.execute(text("ALTER TABLE reports ADD COLUMN ai_analysis TEXT"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    patch_legacy_schema()
    seed_default_zones()
    asyncio.create_task(sensors.read_arduino_serial())
    yield


app = FastAPI(title="건설현장 관리 API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workers.router)
app.include_router(alerts.router)
app.include_router(sensors.router)
app.include_router(photos.router)
app.include_router(report.router)
app.include_router(weather.router)

uploads_dir = os.path.join(os.path.dirname(__file__), "../uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
