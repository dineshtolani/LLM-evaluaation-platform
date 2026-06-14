import json
import time
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid as uuid_pkg

from app.evaluation.ollama_client import OllamaClient
from app.evaluation.metrics import (
    compute_quality_score,
    compute_relevance,
    compute_hallucination_score,
    compute_factual_consistency,
    compute_sentence_level_hallucination,
    compute_token_cost,
)
from app.evaluation.toxicity import compute_toxicity
from app.models.prompt import Prompt
from app.models.evaluation import Evaluation
from app.models.llm_model import LLMModel
from app.services.mlflow_service import MLflowService


class Evaluator:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.ollama = OllamaClient()
        self.mlflow = MLflowService()

    async def evaluate(
        self,
        prompt_id: uuid_pkg.UUID,
        model_name: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 40,
        num_ctx: int = 4096,
        llm_model_id: Optional[uuid_pkg.UUID] = None,
        use_gpu: bool = True,
    ) -> Evaluation:
        prompt_result = await self.db.execute(
            select(Prompt).where(Prompt.id == prompt_id)
        )
        prompt_obj = prompt_result.scalar_one_or_none()
        if not prompt_obj:
            raise ValueError(f"Prompt {prompt_id} not found")

        if llm_model_id:
            model_result = await self.db.execute(
                select(LLMModel).where(LLMModel.id == llm_model_id)
            )
            model_obj = model_result.scalar_one_or_none()
            if model_obj:
                model_name = model_obj.name

        model_result = await self.db.execute(
            select(LLMModel).where(LLMModel.name == model_name)
        )
        model_obj = model_result.scalar_one_or_none()
        if not model_obj:
            model_obj = LLMModel(
                id=uuid_pkg.uuid4(),
                name=model_name,
                provider="ollama",
                model_type="open-source",
                gpu_required=use_gpu,
            )
            self.db.add(model_obj)
            await self.db.flush()

        llm_model_id = model_obj.id

        self.mlflow.start_run(prompt_obj.name, prompt_obj.id, llm_model_id)

        start_time = time.time()

        result = await self.ollama.generate(
            model=model_name,
            prompt=prompt_obj.content,
            system_prompt=prompt_obj.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            num_ctx=num_ctx,
            use_gpu=use_gpu,
        )

        latency_ms = result["latency_ms"]
        response_text = result["response"]
        prompt_tokens = result["prompt_tokens"]
        completion_tokens = result["completion_tokens"]
        total_tokens = result["total_tokens"]

        gpu_info = await self.ollama.get_gpu_info()

        cost = compute_token_cost(
            prompt_tokens,
            completion_tokens,
            model_obj.cost_per_prompt_token,
            model_obj.cost_per_completion_token,
        )

        hallucination = compute_hallucination_score(
            response_text, prompt_obj.content, prompt_obj.expected_output
        )
        quality = compute_quality_score(response_text)
        relevance = compute_relevance(prompt_obj.content, response_text)
        factual_consistency = compute_factual_consistency(
            response_text, prompt_obj.expected_output
        )

        nli_hallucination = None
        sentence_analysis = None
        if prompt_obj.expected_output:
            nli_hallucination, sentence_analysis = compute_sentence_level_hallucination(
                response_text, prompt_obj.expected_output
            )

        toxicity_result = compute_toxicity(response_text)

        params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "num_ctx": num_ctx,
            "use_gpu": use_gpu,
        }

        evaluation = Evaluation(
            id=uuid_pkg.uuid4(),
            prompt_id=prompt_id,
            llm_model_id=llm_model_id,
            response=response_text,
            latency_ms=latency_ms,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_cost=cost,
            hallucination_score=hallucination,
            nli_hallucination_score=nli_hallucination,
            toxicity_score=toxicity_result["toxicity_score"],
            toxicity_categories_json=json.dumps(toxicity_result["categories"]),
            is_toxic=toxicity_result["is_toxic"],
            quality_score=quality,
            relevance_score=relevance,
            factual_consistency=factual_consistency,
            sentence_analysis_json=json.dumps(sentence_analysis) if sentence_analysis else None,
            gpu_utilization=gpu_info.get("gpu_utilization"),
            gpu_memory_used_mb=gpu_info.get("memory_used_mb"),
            params_json=json.dumps(params),
        )

        self.db.add(evaluation)
        await self.db.commit()
        await self.db.refresh(evaluation)

        self.mlflow.log_evaluation_metrics(
            latency_ms=latency_ms,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_cost=cost,
            hallucination_score=hallucination,
            quality_score=quality,
            relevance_score=relevance,
            factual_consistency=factual_consistency,
            gpu_utilization=gpu_info.get("gpu_utilization"),
        )

        self.mlflow.end_run()

        evaluation.mlflow_run_id = self.mlflow.active_run_id
        await self.db.commit()

        return evaluation

    async def close(self):
        await self.ollama.close()
