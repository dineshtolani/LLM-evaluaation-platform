import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
from app.models.prompt import PromptCategory, PromptStatus


class PromptCreate(BaseModel):
    name: str
    content: str
    system_prompt: Optional[str] = None
    category: PromptCategory = PromptCategory.other
    expected_output: Optional[str] = None
    tags: Optional[str] = None
    metadata_json: Optional[str] = None


class PromptUpdate(BaseModel):
    name: Optional[str] = None
    content: Optional[str] = None
    system_prompt: Optional[str] = None
    category: Optional[PromptCategory] = None
    status: Optional[PromptStatus] = None
    expected_output: Optional[str] = None
    tags: Optional[str] = None
    metadata_json: Optional[str] = None


class PromptResponse(BaseModel):
    id: uuid.UUID
    name: str
    content: str
    system_prompt: Optional[str] = None
    category: PromptCategory
    status: PromptStatus
    version: int
    expected_output: Optional[str] = None
    tags: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PromptList(BaseModel):
    total: int
    items: list[PromptResponse]
