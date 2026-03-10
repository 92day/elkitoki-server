from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import get_db
from models.models import Worker

router = APIRouter(prefix='/api/workers', tags=['workers'])


class WorkerCreate(BaseModel):
    name: str
    role: Optional[str] = None
    phone: Optional[str] = None
    zone_id: Optional[int] = None
    status: Optional[str] = 'work'


class WorkerUpdate(BaseModel):
    role: Optional[str] = None
    zone_id: Optional[int] = None
    status: Optional[str] = None
    phone: Optional[str] = None


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
    worker = Worker(**data.model_dump())
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

    for key, value in data.model_dump(exclude_none=True).items():
        setattr(worker, key, value)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail='Invalid zone_id. Check the zones table.')

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
