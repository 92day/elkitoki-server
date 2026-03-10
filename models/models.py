from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


class Worker(Base):
    __tablename__ = 'workers'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    role = Column(String(50))
    phone = Column(String(20))
    zone_id = Column(Integer, ForeignKey('zones.id'), nullable=True)
    status = Column(String(20), default='work')
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    zone = relationship('Zone', back_populates='workers')


class Zone(Base):
    __tablename__ = 'zones'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    description = Column(String(100))
    task = Column(String(100))
    risk_level = Column(String(20), default='safe')
    max_workers = Column(Integer, default=30)

    workers = relationship('Worker', back_populates='zone')
    sensors = relationship('SensorData', back_populates='zone')
    photos = relationship('Photo', back_populates='zone')


class SensorData(Base):
    __tablename__ = 'sensor_data'

    id = Column(Integer, primary_key=True, index=True)
    zone_id = Column(Integer, ForeignKey('zones.id'), nullable=True)
    sensor_type = Column(String(50))
    value = Column(Float)
    unit = Column(String(20))
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    zone = relationship('Zone', back_populates='sensors')


class Alert(Base):
    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(20))
    message = Column(Text)
    source = Column(String(50))
    is_resolved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Photo(Base):
    __tablename__ = 'photos'

    id = Column(Integer, primary_key=True, index=True)
    zone_id = Column(Integer, ForeignKey('zones.id'), nullable=True)
    file_path = Column(String(300))
    original_name = Column(String(200))
    ai_result = Column(Text)
    risk_detected = Column(Boolean, default=False)
    taken_at = Column(DateTime(timezone=True), server_default=func.now())

    zone = relationship('Zone', back_populates='photos')


class Report(Base):
    __tablename__ = 'reports'

    id = Column(Integer, primary_key=True, index=True)
    date = Column(String(20))
    text_content = Column(Text)
    translated_text = Column(Text)
    source_language = Column(String(10), default='ko')
    target_language = Column(String(10), nullable=True)
    author_name = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Progress(Base):
    __tablename__ = 'progress'

    id = Column(Integer, primary_key=True, index=True)
    task_name = Column(String(100))
    percentage = Column(Integer, default=0)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())


class TranslationLog(Base):
    __tablename__ = 'translation_logs'

    id = Column(Integer, primary_key=True, index=True)
    source_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    source_language = Column(String(10), nullable=False)
    target_language = Column(String(10), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
