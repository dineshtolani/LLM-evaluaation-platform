from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, Integer
from typing import Optional
import uuid
import json

from app.database import get_db
from app.models.evaluation import Evaluation
from app.models.prompt import Prompt
from app.models.llm_model import LLMModel
from app.schemas.evaluation import EvalRequest, EvalResponse, EvaluationResponse, EvaluationList
from app.evaluation.evaluator import Evaluator
from app.evaluation.toxicity import compute_toxicity
from app.services.alert_service import AlertService
from app.models.alert import AlertMetric

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


@router.post("", response_model=EvalResponse, status_code=201)
async def create_evaluation(req: EvalRequest, db: AsyncSession = Depends(get_db)):
    evaluator = Evaluator(db)
    try:
        eval_result = await evaluator.evaluate(
            prompt_id=req.prompt_id,
            model_name=req.model_name,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            top_k=req.top_k,
            num_ctx=req.num_ctx,
            llm_model_id=req.llm_model_id,
            use_gpu=req.use_gpu,
            use_deepeval=req.use_deepeval,
        )

        alert_service = AlertService(db)
        await alert_service.check_alerts(
            AlertMetric.latency, eval_result.latency_ms,
            prompt_id=req.prompt_id, llm_model_id=eval_result.llm_model_id,
        )
        if eval_result.hallucination_score is not None:
            await alert_service.check_alerts(
                AlertMetric.hallucination, eval_result.hallucination_score,
                prompt_id=req.prompt_id, llm_model_id=eval_result.llm_model_id,
            )
        await alert_service.check_alerts(
            AlertMetric.cost, eval_result.token_cost,
            prompt_id=req.prompt_id, llm_model_id=eval_result.llm_model_id,
        )

        return EvalResponse(
            id=eval_result.id,
            prompt_id=eval_result.prompt_id,
            llm_model_id=eval_result.llm_model_id,
            model_name=req.model_name,
            response=eval_result.response,
            latency_ms=eval_result.latency_ms,
            total_tokens=eval_result.total_tokens,
            prompt_tokens=eval_result.prompt_tokens,
            completion_tokens=eval_result.completion_tokens,
            token_cost=eval_result.token_cost,
            hallucination_score=eval_result.hallucination_score,
            toxicity_score=eval_result.toxicity_score,
            is_toxic=eval_result.is_toxic,
            quality_score=eval_result.quality_score,
            relevance_score=eval_result.relevance_score,
            factual_consistency=eval_result.factual_consistency,
            deepeval_faithfulness_score=eval_result.deepeval_faithfulness_score,
            deepeval_hallucination_score=eval_result.deepeval_hallucination_score,
            deepeval_toxicity_score=eval_result.deepeval_toxicity_score,
            deepeval_bias_score=eval_result.deepeval_bias_score,
            deepeval_g_eval_score=eval_result.deepeval_g_eval_score,
            gpu_utilization=eval_result.gpu_utilization,
            gpu_memory_used_mb=eval_result.gpu_memory_used_mb,
            mlflow_run_id=eval_result.mlflow_run_id,
            created_at=eval_result.created_at,
        )
    finally:
        await evaluator.close()


@router.get("", response_model=EvaluationList)
async def list_evaluations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    prompt_id: Optional[uuid.UUID] = None,
    model_name: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(
        Evaluation,
        Prompt.name.label("prompt_name"),
        LLMModel.name.label("model_name"),
    ).join(Prompt, Evaluation.prompt_id == Prompt.id).join(LLMModel, Evaluation.llm_model_id == LLMModel.id)

    if prompt_id:
        query = query.where(Evaluation.prompt_id == prompt_id)
    if model_name:
        query = query.where(LLMModel.name == model_name)

    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    query = query.order_by(desc(Evaluation.created_at)).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    rows = result.all()

    items = []
    for eval_row, pname, mname in rows:
        items.append(EvaluationResponse(
            id=eval_row.id,
            prompt_id=eval_row.prompt_id,
            llm_model_id=eval_row.llm_model_id,
            response=eval_row.response,
            latency_ms=eval_row.latency_ms,
            total_tokens=eval_row.total_tokens,
            prompt_tokens=eval_row.prompt_tokens,
            completion_tokens=eval_row.completion_tokens,
            token_cost=eval_row.token_cost,
            hallucination_score=eval_row.hallucination_score,
            toxicity_score=eval_row.toxicity_score,
            is_toxic=eval_row.is_toxic,
            quality_score=eval_row.quality_score,
            relevance_score=eval_row.relevance_score,
            factual_consistency=eval_row.factual_consistency,
            deepeval_faithfulness_score=eval_row.deepeval_faithfulness_score,
            deepeval_hallucination_score=eval_row.deepeval_hallucination_score,
            deepeval_toxicity_score=eval_row.deepeval_toxicity_score,
            deepeval_bias_score=eval_row.deepeval_bias_score,
            deepeval_g_eval_score=eval_row.deepeval_g_eval_score,
            gpu_utilization=eval_row.gpu_utilization,
            gpu_memory_used_mb=eval_row.gpu_memory_used_mb,
            mlflow_run_id=eval_row.mlflow_run_id,
            params_json=eval_row.params_json,
            error_message=eval_row.error_message,
            created_at=eval_row.created_at,
            prompt_name=pname,
            model_name=mname,
        ))

    return EvaluationList(total=total, items=items)


@router.get("/stats")
async def get_evaluation_stats(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(
            func.avg(Evaluation.latency_ms).label("avg_latency"),
            func.avg(Evaluation.hallucination_score).label("avg_hallucination"),
            func.avg(Evaluation.quality_score).label("avg_quality"),
            func.avg(Evaluation.token_cost).label("avg_cost"),
            func.avg(Evaluation.toxicity_score).label("avg_toxicity"),
            func.sum(Evaluation.total_tokens).label("total_tokens"),
            func.count(Evaluation.id).label("total_evaluations"),
            func.sum(Evaluation.is_toxic.cast(Integer)).label("toxic_count"),
        )
    )
    stats = result.one()
    return {
        "avg_latency_ms": round(float(stats.avg_latency or 0), 2),
        "avg_hallucination_score": round(float(stats.avg_hallucination or 0), 4),
        "avg_quality_score": round(float(stats.avg_quality or 0), 4),
        "avg_cost_per_eval": round(float(stats.avg_cost or 0), 6),
        "avg_toxicity_score": round(float(stats.avg_toxicity or 0), 4),
        "total_tokens_consumed": int(stats.total_tokens or 0),
        "total_evaluations": int(stats.total_evaluations or 0),
        "toxic_responses": int(stats.toxic_count or 0),
    }


from pydantic import BaseModel  # noqa: E402


class DetoxRequest(BaseModel):
    text: str
    threshold: float = 0.5


class DetoxResponse(BaseModel):
    original_text: str
    toxicity_score: float
    categories: dict
    is_toxic: bool
    is_blocked: bool
    message: str


class BatchEvalRequest(BaseModel):
    prompt_ids: list[uuid.UUID]
    model_name: str = "tinyllama"
    temperature: float = 0.7
    max_tokens: int = 512


@router.post("/detox", response_model=DetoxResponse)
async def check_toxicity(req: DetoxRequest):
    result = compute_toxicity(req.text)
    is_blocked = result["toxicity_score"] > req.threshold
    return DetoxResponse(
        original_text=req.text,
        toxicity_score=result["toxicity_score"],
        categories=result["categories"],
        is_toxic=result["is_toxic"],
        is_blocked=is_blocked,
        message="Content blocked due to toxicity" if is_blocked else "Content approved",
    )


@router.post("/batch", response_model=list[EvalResponse])
async def batch_evaluate(req: BatchEvalRequest, db: AsyncSession = Depends(get_db)):
    results = []
    for prompt_id in req.prompt_ids:
        evaluator = Evaluator(db)
        try:
            eval_result = await evaluator.evaluate(
                prompt_id=prompt_id,
                model_name=req.model_name,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            results.append(EvalResponse(
                id=eval_result.id,
                prompt_id=eval_result.prompt_id,
                llm_model_id=eval_result.llm_model_id,
                model_name=req.model_name,
                response=eval_result.response,
                latency_ms=eval_result.latency_ms,
                total_tokens=eval_result.total_tokens,
                prompt_tokens=eval_result.prompt_tokens,
                completion_tokens=eval_result.completion_tokens,
                token_cost=eval_result.token_cost,
                hallucination_score=eval_result.hallucination_score,
                toxicity_score=eval_result.toxicity_score,
                is_toxic=eval_result.is_toxic,
                quality_score=eval_result.quality_score,
                relevance_score=eval_result.relevance_score,
                factual_consistency=eval_result.factual_consistency,
                deepeval_faithfulness_score=eval_result.deepeval_faithfulness_score,
                deepeval_hallucination_score=eval_result.deepeval_hallucination_score,
                deepeval_toxicity_score=eval_result.deepeval_toxicity_score,
                deepeval_bias_score=eval_result.deepeval_bias_score,
                deepeval_g_eval_score=eval_result.deepeval_g_eval_score,
                gpu_utilization=eval_result.gpu_utilization,
                gpu_memory_used_mb=eval_result.gpu_memory_used_mb,
                mlflow_run_id=eval_result.mlflow_run_id,
                created_at=eval_result.created_at,
            ))
        finally:
            await evaluator.close()
    return results


class ToxicityReportResponse(BaseModel):
    total_evaluations: int
    toxic_count: int
    toxic_percentage: float
    avg_toxicity: float
    max_toxicity: float
    toxic_by_category: dict
    most_toxic: list[dict]


@router.get("/toxicity-report")
async def toxicity_report(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(
            func.count(Evaluation.id).label("total"),
            func.sum(Evaluation.is_toxic.cast(Integer)).label("toxic"),
            func.avg(Evaluation.toxicity_score).label("avg_tox"),
            func.max(Evaluation.toxicity_score).label("max_tox"),
        )
    )
    stats = result.one()
    total = int(stats.total or 0)
    toxic = int(stats.toxic or 0)

    toxic_evals = await db.execute(
        select(
            Evaluation.id,
            Evaluation.response,
            Evaluation.toxicity_score,
            Evaluation.toxicity_categories_json,
            Evaluation.is_toxic,
            Evaluation.created_at,
            Prompt.name.label("prompt_name"),
            LLMModel.name.label("model_name"),
        )
        .join(Prompt, Evaluation.prompt_id == Prompt.id)
        .join(LLMModel, Evaluation.llm_model_id == LLMModel.id)
        .where(Evaluation.toxicity_score > 0)
        .order_by(desc(Evaluation.toxicity_score))
        .limit(20)
    )
    rows = toxic_evals.all()

    most_toxic = []
    cat_counts = {}
    for row in rows:
        entry = {
            "id": str(row.id),
            "prompt_name": row.prompt_name,
            "model_name": row.model_name,
            "toxicity_score": round(float(row.toxicity_score or 0), 4),
            "is_toxic": bool(row.is_toxic),
            "response_preview": (row.response or "")[:100],
            "created_at": str(row.created_at),
        }
        most_toxic.append(entry)

        if row.toxicity_categories_json:
            try:
                cats = json.loads(row.toxicity_categories_json)
                for cat, score in cats.items():
                    if score > 0:
                        cat_counts[cat] = cat_counts.get(cat, 0) + 1
            except Exception:
                pass

    return ToxicityReportResponse(
        total_evaluations=total,
        toxic_count=toxic,
        toxic_percentage=round((toxic / total * 100) if total > 0 else 0, 2),
        avg_toxicity=round(float(stats.avg_tox or 0), 4),
        max_toxicity=round(float(stats.max_tox or 0), 4),
        toxic_by_category=cat_counts,
        most_toxic=most_toxic,
    )


class HallucinationReportResponse(BaseModel):
    total_evaluations: int
    avg_hallucination_score: float
    max_hallucination_score: float
    min_hallucination_score: float
    high_hallucination_count: int
    high_hallucination_percentage: float
    high_hallucination_threshold: float
    most_hallucinated: list[dict]


@router.get("/hallucination-report")
async def hallucination_report(
    threshold: float = Query(0.3, description="High hallucination threshold"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(
            func.count(Evaluation.id).label("total"),
            func.avg(Evaluation.hallucination_score).label("avg_hal"),
            func.max(Evaluation.hallucination_score).label("max_hal"),
            func.min(Evaluation.hallucination_score).label("min_hal"),
        )
        .where(Evaluation.hallucination_score.isnot(None))
    )
    stats = result.one()
    total = int(stats.total or 0)

    high_hal = await db.execute(
        select(func.count(Evaluation.id))
        .where(
            Evaluation.hallucination_score.isnot(None),
            Evaluation.hallucination_score > threshold,
        )
    )
    high_count = int(high_hal.scalar() or 0)

    hal_evals = await db.execute(
        select(
            Evaluation.id,
            Evaluation.response,
            Evaluation.hallucination_score,
            Evaluation.quality_score,
            Evaluation.latency_ms,
            Evaluation.total_tokens,
            Evaluation.created_at,
            Prompt.name.label("prompt_name"),
            Prompt.content.label("prompt_content"),
            LLMModel.name.label("model_name"),
        )
        .join(Prompt, Evaluation.prompt_id == Prompt.id)
        .join(LLMModel, Evaluation.llm_model_id == LLMModel.id)
        .where(Evaluation.hallucination_score.isnot(None))
        .order_by(desc(Evaluation.hallucination_score))
        .limit(limit)
    )
    rows = hal_evals.all()

    most_hallucinated = [
        {
            "id": str(row.id),
            "prompt_name": row.prompt_name,
            "prompt_content": (row.prompt_content or "")[:100],
            "model_name": row.model_name,
            "hallucination_score": round(float(row.hallucination_score or 0), 4),
            "quality_score": round(float(row.quality_score or 0), 4),
            "latency_ms": round(float(row.latency_ms or 0), 2),
            "tokens": int(row.total_tokens or 0),
            "response_preview": (row.response or "")[:100],
            "created_at": str(row.created_at),
        }
        for row in rows
    ]

    return HallucinationReportResponse(
        total_evaluations=total,
        avg_hallucination_score=round(float(stats.avg_hal or 0), 4),
        max_hallucination_score=round(float(stats.max_hal or 0), 4),
        min_hallucination_score=round(float(stats.min_hal or 0), 4),
        high_hallucination_count=high_count,
        high_hallucination_percentage=round((high_count / total * 100) if total > 0 else 0, 2),
        high_hallucination_threshold=threshold,
        most_hallucinated=most_hallucinated,
    )
