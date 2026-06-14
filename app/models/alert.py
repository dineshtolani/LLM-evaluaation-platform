import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Float, Boolean, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base
import enum


class AlertMetric(str, enum.Enum):
    latency = "latency"
    hallucination = "hallucination"
    cost = "cost"
    quality = "quality"
    token_usage = "token_usage"


class AlertOperator(str, enum.Enum):
    gt = "gt"
    lt = "lt"
    gte = "gte"
    lte = "lte"
    eq = "eq"


class AlertStatus(str, enum.Enum):
    active = "active"
    triggered = "triggered"
    muted = "muted"


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    metric: Mapped[AlertMetric] = mapped_column(SAEnum(AlertMetric), nullable=False)
    operator: Mapped[AlertOperator] = mapped_column(SAEnum(AlertOperator), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    llm_model_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    prompt_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    status: Mapped[AlertStatus] = mapped_column(SAEnum(AlertStatus), default=AlertStatus.active)
    last_triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notification_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
