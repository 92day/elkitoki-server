import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text

from database import SessionLocal, engine
from models.models import Base, Zone
from routers import alerts, photos, report, sensors, translations, weather, workers


def seed_default_zones() -> None:
    db = SessionLocal()
    try:
        defaults = [
            {'id': 1, 'name': 'Zone A', 'description': 'B2', 'task': 'Rebar Work', 'risk_level': 'safe'},
            {'id': 2, 'name': 'Zone B', 'description': 'B1', 'task': 'Concrete', 'risk_level': 'safe'},
            {'id': 3, 'name': 'Zone C', 'description': '1F-3F', 'task': 'High-altitude Work', 'risk_level': 'caution'},
            {'id': 4, 'name': 'Zone D', 'description': '4F-6F', 'task': 'Frame Construction', 'risk_level': 'safe'},
            {'id': 5, 'name': 'Zone E', 'description': 'Roof', 'task': 'Roof Work', 'risk_level': 'danger'},
            {'id': 6, 'name': 'Zone F', 'description': 'Exterior', 'task': 'Facade Finishing', 'risk_level': 'safe'},
        ]

        existing_ids = {row[0] for row in db.query(Zone.id).filter(Zone.id.in_([1, 2, 3, 4, 5, 6])).all()}
        for zone in defaults:
            if zone['id'] not in existing_ids:
                db.add(Zone(**zone))

        db.commit()
    finally:
        db.close()


def patch_legacy_schema() -> None:
    with engine.begin() as conn:
        inspector = inspect(conn)
        table_names = set(inspector.get_table_names())

        if 'photos' in table_names:
            photo_columns = {column['name'] for column in inspector.get_columns('photos')}
            if 'risk_detected' not in photo_columns:
                conn.execute(text('ALTER TABLE photos ADD COLUMN risk_detected BOOLEAN DEFAULT 0'))

        if 'reports' in table_names:
            report_columns = {column['name'] for column in inspector.get_columns('reports')}
            if 'translated_text' not in report_columns:
                conn.execute(text('ALTER TABLE reports ADD COLUMN translated_text TEXT NULL'))
            if 'source_language' not in report_columns:
                conn.execute(text("ALTER TABLE reports ADD COLUMN source_language VARCHAR(10) DEFAULT 'ko'"))
            if 'target_language' not in report_columns:
                conn.execute(text('ALTER TABLE reports ADD COLUMN target_language VARCHAR(10) NULL'))


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    patch_legacy_schema()
    seed_default_zones()

    if os.getenv('ENABLE_ARDUINO', '0') == '1':
        asyncio.create_task(sensors.read_arduino_serial())

    yield


app = FastAPI(title='Elkitoki Site API', version='2.3', lifespan=lifespan)


@app.get('/health')
def health():
    return {'status': 'ok'}


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(workers.router)
app.include_router(alerts.router)
app.include_router(sensors.router)
app.include_router(photos.router)
app.include_router(report.router)
app.include_router(translations.router)
app.include_router(weather.router)

base_dir = os.path.dirname(os.path.abspath(__file__))
uploads_dir = os.path.join(base_dir, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(os.path.join(uploads_dir, 'audio'), exist_ok=True)
os.makedirs(os.path.join(uploads_dir, 'photos'), exist_ok=True)
app.mount('/uploads', StaticFiles(directory=uploads_dir), name='uploads')
