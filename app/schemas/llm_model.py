import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class LLMModelCreate(BaseModel):
    name: str
    provider: str = "ollama"
    description: Optional[str] = None
    model_type: str = "open-source"
    context_window: int = 4096
    is_active: bool = True
    gpu_required: bool = True
    vram_required_mb: Optional[int] = None
    default_params: Optional[str] = None
    cost_per_prompt_token: float = 0.000003
    cost_per_completion_token: float = 0.000015


class LLMModelResponse(BaseModel):
    id: uuid.UUID
    name: str
    provider: str
    description: Optional[str] = None
    model_type: str
    context_window: int
    is_active: bool
    gpu_required: bool
    vram_required_mb: Optional[int] = None
    default_params: Optional[str] = None
    cost_per_prompt_token: float
    cost_per_completion_token: float
    created_at: datetime

    model_config = {"from_attributes": True}


class LLMModelList(BaseModel):
    total: int
    items: list[LLMModelResponse]
