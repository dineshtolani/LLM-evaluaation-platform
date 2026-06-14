import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Enum as SAEnum, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base
import enum


class PromptStatus(str, enum.Enum):
    active = "active"
    archived = "archived"
    deprecated = "deprecated"


class PromptCategory(str, enum.Enum):
    qa = "qa"
    summarization = "summarization"
    classification = "classification"
    extraction = "extraction"
    generation = "generation"
    reasoning = "reasoning"
    code = "code"
    other = "other"


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[PromptCategory] = mapped_column(SAEnum(PromptCategory), default=PromptCategory.other)
    status: Mapped[PromptStatus] = mapped_column(SAEnum(PromptStatus), default=PromptStatus.active)
    version: Mapped[int] = mapped_column(default=1)
    expected_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    evaluations = relationship("Evaluation", back_populates="prompt", cascade="all, delete-orphan")
