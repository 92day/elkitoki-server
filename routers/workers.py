from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import get_db
from models.models import Worker

router = APIRouter(prefix='/api/workers', tags=['workers'])

WORKER_ROLE_OPTIONS = [
    '소장',
    '안전관리자',
    '현장관리자',
    '현장작업자',
    '기타',
]


class WorkerCreate(BaseModel):
    name: str
    role: Optional[str] = None
    phone: Optional[str] = None
    zone_id: Optional[int] = None
    status: Optional[str] = 'work'
    heart_rate: Optional[int] = None
    shift_started_at: Optional[datetime] = None

    @field_validator('role', mode='before')
    @classmethod
    def validate_role(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned not in WORKER_ROLE_OPTIONS:
            raise ValueError('invalid worker role')
        return cleaned


class WorkerUpdate(BaseModel):
    role: Optional[str] = None
    zone_id: Optional[int] = None
    status: Optional[str] = None
    phone: Optional[str] = None
    heart_rate: Optional[int] = None
    shift_started_at: Optional[datetime] = None

    @field_validator('role', mode='before')
    @classmethod
    def validate_role(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned not in WORKER_ROLE_OPTIONS:
            raise ValueError('invalid worker role')
        return cleaned


class WorkerVitalsUpdate(BaseModel):
    heart_rate: Optional[int] = None
    shift_started_at: Optional[datetime] = None


@router.get('/roles')
def get_worker_roles():
    return {'roles': WORKER_ROLE_OPTIONS}


@router.get('/')
def get_workers(db: Session = Depends(get_db)):
    return db.query(Worker).all()


@router.get('/{worker_id}')
def get_worker(worker_id: int, db: Session = Depends(get_db)):
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail='Worker not found.')
    return worker


@router.post('/')
def create_worker(data: WorkerCreate, db: Session = Depends(get_db)):
    payload = data.model_dump()
    if payload.get('status') == 'work' and not payload.get('shift_started_at'):
        payload['shift_started_at'] = datetime.now(timezone.utc)

    worker = Worker(**payload)
    db.add(worker)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail='Invalid zone_id. Check the zones table.')
    db.refresh(worker)
    return worker


@router.patch('/{worker_id}')
def update_worker(worker_id: int, data: WorkerUpdate, db: Session = Depends(get_db)):
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail='Worker not found.')

    updates = data.model_dump(exclude_none=True)

    if updates.get('status') == 'work' and worker.shift_started_at is None and 'shift_started_at' not in updates:
        updates['shift_started_at'] = datetime.now(timezone.utc)

    for key, value in updates.items():
        setattr(worker, key, value)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail='Invalid zone_id. Check the zones table.')

    db.refresh(worker)
    return worker


@router.patch('/{worker_id}/vitals')
def update_worker_vitals(worker_id: int, data: WorkerVitalsUpdate, db: Session = Depends(get_db)):
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail='Worker not found.')

    updates = data.model_dump(exclude_none=True)
    for key, value in updates.items():
        setattr(worker, key, value)

    db.commit()
    db.refresh(worker)
    return worker


@router.delete('/{worker_id}')
def delete_worker(worker_id: int, db: Session = Depends(get_db)):
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail='Worker not found.')

    db.delete(worker)
    db.commit()
    return {'message': 'Deleted'}



