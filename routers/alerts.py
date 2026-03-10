from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models.models import Alert

router = APIRouter(prefix='/api/alerts', tags=['alerts'])


class AlertCreate(BaseModel):
    level: str
    message: str
    source: Optional[str] = 'Manual Input'


@router.get('/')
def get_alerts(db: Session = Depends(get_db)):
    return db.query(Alert).filter(Alert.is_resolved.is_(False)).order_by(Alert.created_at.desc()).all()


@router.post('/')
def create_alert(data: AlertCreate, db: Session = Depends(get_db)):
    alert = Alert(**data.model_dump())
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
