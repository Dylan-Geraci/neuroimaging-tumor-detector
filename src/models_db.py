"""
SQLAlchemy ORM models for prediction history.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, DateTime, JSON
from src.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    filename = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)
    batch_id = Column(String, nullable=True)
