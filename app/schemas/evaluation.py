import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class EvalRequest(BaseModel):
    prompt_id: uuid.UUID
    llm_model_id: Optional[uuid.UUID] = None
    model_name: Optional[str] = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 40
    num_ctx: int = 4096
    use_gpu: bool = True
    use_deepeval: bool = False


class EvalResponse(BaseModel):
    id: uuid.UUID
    prompt_id: uuid.UUID
    llm_model_id: uuid.UUID
    model_name: str
    response: str
    latency_ms: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    token_cost: float
    hallucination_score: Optional[float] = None
    toxicity_score: Optional[float] = None
    is_toxic: Optional[bool] = None
    quality_score: Optional[float] = None
    relevance_score: Optional[float] = None
    factual_consistency: Optional[float] = None
    deepeval_faithfulness_score: Optional[float] = None
    deepeval_hallucination_score: Optional[float] = None
    deepeval_toxicity_score: Optional[float] = None
    deepeval_bias_score: Optional[float] = None
    deepeval_g_eval_score: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    mlflow_run_id: Optional[str] = None
    created_at: datetime


class EvaluationResponse(BaseModel):
    id: uuid.UUID
    prompt_id: uuid.UUID
    llm_model_id: uuid.UUID
    response: str
    latency_ms: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    token_cost: float
    hallucination_score: Optional[float] = None
    toxicity_score: Optional[float] = None
    is_toxic: Optional[bool] = None
    quality_score: Optional[float] = None
    relevance_score: Optional[float] = None
    factual_consistency: Optional[float] = None
    deepeval_faithfulness_score: Optional[float] = None
    deepeval_hallucination_score: Optional[float] = None
    deepeval_toxicity_score: Optional[float] = None
    deepeval_bias_score: Optional[float] = None
    deepeval_g_eval_score: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    mlflow_run_id: Optional[str] = None
    params_json: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    prompt_name: Optional[str] = None
    model_name: Optional[str] = None

    model_config = {"from_attributes": True}


class EvaluationList(BaseModel):
    total: int
    items: list[EvaluationResponse]
