import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text
import uvicorn

from database import SessionLocal, engine
from models.models import Alert, Base, Photo, SensorData, User, Worker, Zone
from routers import alerts, auth, photos, report, translations, weather, workers

class QuietSensorAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        noisy_paths = ('/api/sensors/events', '/api/device/commands/pending')
        return not any(path in message for path in noisy_paths)

logging.getLogger('uvicorn.access').addFilter(QuietSensorAccessFilter())



def map_user_role_to_worker_role(user_role: str | None) -> str:
    if not user_role:
        return '\uAE30\uD0C0'

    role_map = {
        'site_manager': '\uC18C\uC7A5',
        '\uC18C\uC7A5': '\uC18C\uC7A5',
        '\uC548\uC804\uAD00\uB9AC\uC790': '\uC548\uC804\uAD00\uB9AC\uC790',
        '\uD604\uC7A5\uAD00\uB9AC\uC790': '\uD604\uC7A5\uAD00\uB9AC\uC790',
        '\uD604\uC7A5\uC791\uC5C5\uC790': '\uD604\uC7A5\uC791\uC5C5\uC790',
        '\uAE30\uD0C0': '\uAE30\uD0C0',
    }
    return role_map.get(user_role, '\uAE30\uD0C0')


def seed_default_zones() -> None:
    auth.seed_default_admin()
    db = SessionLocal()
    try:
        defaults = [
            {'id': 1, 'name': 'Zone A', 'description': 'B2', 'task': 'Rebar Work', 'risk_level': 'safe'},
            {'id': 2, 'name': 'Zone B', 'description': 'B1', 'task': 'Concrete', 'risk_level': 'safe'},
            {'id': 3, 'name': 'Zone C', 'description': '1F-3F', 'task': 'High-altitude Work', 'risk_level': 'caution'},
        ]
        deprecated_zone_ids = [4, 5, 6]

        db.query(Worker).filter(Worker.zone_id.in_(deprecated_zone_ids)).update({Worker.zone_id: None}, synchronize_session=False)
        db.query(SensorData).filter(SensorData.zone_id.in_(deprecated_zone_ids)).update({SensorData.zone_id: None}, synchronize_session=False)
        db.query(Photo).filter(Photo.zone_id.in_(deprecated_zone_ids)).update({Photo.zone_id: None}, synchronize_session=False)
        db.query(Alert).filter(Alert.zone_id.in_(deprecated_zone_ids)).update(
            {Alert.zone_id: None, Alert.zone_name: None},
            synchronize_session=False,
        )
        db.query(Zone).filter(Zone.id.in_(deprecated_zone_ids)).delete(synchronize_session=False)

        existing_ids = {row[0] for row in db.query(Zone.id).filter(Zone.id.in_([1, 2, 3])).all()}
        for zone in defaults:
            if zone['id'] not in existing_ids:
                db.add(Zone(**zone))
            else:
                db.query(Zone).filter(Zone.id == zone['id']).update(zone, synchronize_session=False)

        db.commit()
    finally:
        db.close()



def seed_default_workers() -> None:
    db = SessionLocal()
    try:
        users = db.query(User).filter(User.is_active.is_(True)).all()
        for user in users:
            worker = db.query(Worker).filter(Worker.user_id == user.id).first()
            if not worker:
                worker = db.query(Worker).filter(Worker.name == user.name).first()

            worker_role = map_user_role_to_worker_role(user.role)
            default_phone = '010-0000-0000' if user.username == auth.DEFAULT_ADMIN_USERNAME else None

            if worker:
                worker.user_id = user.id
                worker.name = user.name
                worker.role = worker_role
                if default_phone and not worker.phone:
                    worker.phone = default_phone
                if not worker.status:
                    worker.status = 'work'
                continue

            db.add(
                Worker(
                    user_id=user.id,
                    name=user.name,
                    role=worker_role,
                    phone=default_phone,
                    status='work',
                )
            )
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

        if 'alerts' in table_names:
            alert_columns = {column['name'] for column in inspector.get_columns('alerts')}
            if 'zone_id' not in alert_columns:
                conn.execute(text('ALTER TABLE alerts ADD COLUMN zone_id INTEGER NULL'))
            if 'zone_name' not in alert_columns:
                conn.execute(text('ALTER TABLE alerts ADD COLUMN zone_name VARCHAR(50) NULL'))

        if 'reports' in table_names:
            report_columns = {column['name'] for column in inspector.get_columns('reports')}
            if 'translated_text' not in report_columns:
                conn.execute(text('ALTER TABLE reports ADD COLUMN translated_text TEXT NULL'))
            if 'source_language' not in report_columns:
                conn.execute(text("ALTER TABLE reports ADD COLUMN source_language VARCHAR(10) DEFAULT 'ko'"))
            if 'target_language' not in report_columns:
                conn.execute(text('ALTER TABLE reports ADD COLUMN target_language VARCHAR(10) NULL'))
            if 'entry_type' not in report_columns:
                conn.execute(text("ALTER TABLE reports ADD COLUMN entry_type VARCHAR(20) DEFAULT 'translation'"))

        if 'workers' in table_names:
            worker_columns = {column['name'] for column in inspector.get_columns('workers')}
            if 'user_id' not in worker_columns:
                conn.execute(text('ALTER TABLE workers ADD COLUMN user_id INTEGER NULL'))


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    patch_legacy_schema()
    seed_default_zones()
    seed_default_workers()

    if os.getenv('ENABLE_ARDUINO', '0') == '1':
        asyncio.create_task(alerts.read_arduino_serial())

    yield


app = FastAPI(title='Elkitoki Site API', version='2.5', lifespan=lifespan)


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
app.include_router(alerts.sensor_router)
app.include_router(alerts.device_router)
app.include_router(photos.router)
app.include_router(report.router)
app.include_router(translations.router)
app.include_router(weather.router)
app.include_router(auth.router)

base_dir = os.path.dirname(os.path.abspath(__file__))
uploads_dir = os.path.join(base_dir, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(os.path.join(uploads_dir, 'audio'), exist_ok=True)
os.makedirs(os.path.join(uploads_dir, 'photos'), exist_ok=True)
app.mount('/uploads', StaticFiles(directory=uploads_dir), name='uploads')


if __name__ == '__main__':
    host = os.getenv('UVICORN_HOST', '0.0.0.0')
    port = int(os.getenv('PORT', os.getenv('UVICORN_PORT', '8000')))
    reload_enabled = os.getenv('UVICORN_RELOAD', '1').strip().lower() in {'1', 'true', 'yes', 'on'}
    uvicorn.run('main:app', host=host, port=port, reload=reload_enabled)

