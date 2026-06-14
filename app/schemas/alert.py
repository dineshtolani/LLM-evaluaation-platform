import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
from app.models.alert import AlertMetric, AlertOperator, AlertStatus


class AlertCreate(BaseModel):
    name: str
    metric: AlertMetric
    operator: AlertOperator
    threshold: float
    llm_model_id: Optional[uuid.UUID] = None
    prompt_id: Optional[uuid.UUID] = None
    notification_url: Optional[str] = None


class AlertUpdate(BaseModel):
    name: Optional[str] = None
    threshold: Optional[float] = None
    status: Optional[AlertStatus] = None
    notification_url: Optional[str] = None


class AlertResponse(BaseModel):
    id: uuid.UUID
    name: str
    metric: AlertMetric
    operator: AlertOperator
    threshold: float
    llm_model_id: Optional[uuid.UUID] = None
    prompt_id: Optional[uuid.UUID] = None
    status: AlertStatus
    last_triggered_at: Optional[datetime] = None
    notification_url: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertList(BaseModel):
    total: int
    items: list[AlertResponse]
