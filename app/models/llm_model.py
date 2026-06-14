import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Boolean, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class LLMModel(Base):
    __tablename__ = "llm_models"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    provider: Mapped[str] = mapped_column(String(100), nullable=False, default="ollama")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False, default="open-source")
    context_window: Mapped[int] = mapped_column(default=4096)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    gpu_required: Mapped[bool] = mapped_column(Boolean, default=True)
    vram_required_mb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    default_params: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_per_prompt_token: Mapped[float] = mapped_column(Float, default=0.000003)
    cost_per_completion_token: Mapped[float] = mapped_column(Float, default=0.000015)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    evaluations = relationship("Evaluation", back_populates="llm_model", cascade="all, delete-orphan")
