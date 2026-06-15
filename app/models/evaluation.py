import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Float, Integer, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False, index=True)
    llm_model_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("llm_models.id", ondelete="CASCADE"), nullable=False, index=True)

    response: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    token_cost: Mapped[float] = mapped_column(Float, nullable=False)

    hallucination_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    nli_hallucination_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    toxicity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    toxicity_categories_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_toxic: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    factual_consistency: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentence_analysis_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    gpu_utilization: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_memory_used_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    prompt = relationship("Prompt", back_populates="evaluations")
    llm_model = relationship("LLMModel", back_populates="evaluations")
