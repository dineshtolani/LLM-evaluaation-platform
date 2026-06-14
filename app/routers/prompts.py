from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
import uuid

from app.database import get_db
from app.models.prompt import Prompt
from app.schemas.prompt import PromptCreate, PromptUpdate, PromptResponse, PromptList

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


@router.post("", response_model=PromptResponse, status_code=201)
async def create_prompt(prompt_data: PromptCreate, db: AsyncSession = Depends(get_db)):
    prompt = Prompt(**prompt_data.model_dump())
    db.add(prompt)
    await db.commit()
    await db.refresh(prompt)
    return prompt


@router.get("", response_model=PromptList)
async def list_prompts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Prompt)
    if category:
        query = query.where(Prompt.category == category)
    if status:
        query = query.where(Prompt.status == status)

    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    query = query.offset((page - 1) * page_size).limit(page_size).order_by(Prompt.created_at.desc())
    result = await db.execute(query)
    prompts = result.scalars().all()

    return PromptList(total=total, items=prompts)


@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt


@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: uuid.UUID, prompt_data: PromptUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    for key, value in prompt_data.model_dump(exclude_unset=True).items():
        setattr(prompt, key, value)

    prompt.version += 1
    await db.commit()
    await db.refresh(prompt)
    return prompt


@router.delete("/{prompt_id}", status_code=204)
async def delete_prompt(prompt_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    await db.delete(prompt)
    await db.commit()
