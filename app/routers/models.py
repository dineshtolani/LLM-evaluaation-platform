from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
import uuid

from app.database import get_db
from app.models.llm_model import LLMModel
from app.schemas.llm_model import LLMModelCreate, LLMModelResponse, LLMModelList
from app.evaluation.ollama_client import OllamaClient

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("", response_model=LLMModelResponse, status_code=201)
async def register_model(model_data: LLMModelCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(LLMModel).where(LLMModel.name == model_data.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Model already registered")

    model = LLMModel(**model_data.model_dump())
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


@router.get("", response_model=LLMModelList)
async def list_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    query = select(LLMModel)
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    query = query.offset((page - 1) * page_size).limit(page_size).order_by(LLMModel.created_at.desc())
    result = await db.execute(query)
    models = result.scalars().all()
    return LLMModelList(total=total, items=models)


@router.get("/ollama")
async def list_ollama_models():
    client = OllamaClient()
    try:
        models = await client.list_models()
        return {"models": models}
    finally:
        await client.close()


@router.post("/ollama/pull")
async def pull_ollama_model(model_name: str):
    client = OllamaClient()
    try:
        result = await client.pull_model(model_name)
        return result
    finally:
        await client.close()


@router.get("/{model_id}", response_model=LLMModelResponse)
async def get_model(model_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(LLMModel).where(LLMModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(LLMModel).where(LLMModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    await db.delete(model)
    await db.commit()
