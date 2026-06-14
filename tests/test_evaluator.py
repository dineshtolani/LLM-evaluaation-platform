import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import uuid
from datetime import datetime, timezone

pytestmark = pytest.mark.asyncio(loop_scope="function")

from app.evaluation.evaluator import Evaluator
from app.models.prompt import Prompt
from app.models.llm_model import LLMModel


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def sample_prompt():
    return Prompt(
        id=uuid.uuid4(),
        name="test_prompt",
        content="What is the capital of France?",
        category="qa",
        expected_output="Paris",
    )


@pytest.fixture
def sample_model():
    return LLMModel(
        id=uuid.uuid4(),
        name="tinyllama",
        provider="ollama",
        model_type="open-source",
        cost_per_prompt_token=0.000003,
        cost_per_completion_token=0.000015,
    )


class TestEvaluatorInit:
    async def test_init(self, mock_db):
        evaluator = Evaluator(mock_db)
        assert evaluator.db == mock_db
        assert evaluator.ollama is not None
        assert evaluator.mlflow is not None
        await evaluator.close()


class TestEvaluatorEvaluate:
    @patch("app.evaluation.evaluator.OllamaClient")
    @patch("app.evaluation.evaluator.MLflowService")
    @patch("app.evaluation.evaluator.compute_hallucination_score", return_value=0.05)
    @patch("app.evaluation.evaluator.compute_quality_score", return_value=0.9)
    @patch("app.evaluation.evaluator.compute_relevance", return_value=0.85)
    @patch("app.evaluation.evaluator.compute_factual_consistency", return_value=0.95)
    @patch("app.evaluation.evaluator.compute_sentence_level_hallucination", return_value=(0.1, []))
    @patch("app.evaluation.evaluator.compute_token_cost", return_value=0.0005)
    @patch("app.evaluation.evaluator.compute_toxicity", return_value={
        "toxicity_score": 0.0, "max_category_score": 0.0,
        "categories": {}, "is_toxic": False,
    })
    async def test_evaluate_success(
        self,
        mock_tox, mock_cost, mock_sentence_hal, mock_factual,
        mock_relevance, mock_quality, mock_hallucination,
        mock_mlflow, mock_ollama_cls, mock_db, sample_prompt, sample_model,
    ):
        mock_ollama = AsyncMock()
        mock_ollama.generate.return_value = {
            "response": "Paris is the capital of France.",
            "latency_ms": 150.0,
            "total_tokens": 50,
            "prompt_tokens": 10,
            "completion_tokens": 40,
        }
        mock_ollama.get_gpu_info.return_value = {
            "gpu_available": True, "gpu_name": "RTX A500",
            "gpu_utilization": 0.45, "memory_used_mb": 2048,
        }
        mock_ollama_cls.return_value = mock_ollama

        mock_mlflow.return_value.start_run = AsyncMock()
        mock_mlflow.return_value.log_evaluation_metrics = MagicMock()
        mock_mlflow.return_value.end_run = MagicMock()
        mock_mlflow.return_value.active_run_id = "run_123"

        call_count = 0

        async def execute_side_effect(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = sample_prompt
            else:
                result.scalar_one_or_none.return_value = sample_model
            return result

        mock_db.execute = execute_side_effect
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        evaluator = Evaluator(mock_db)
        result = await evaluator.evaluate(
            prompt_id=sample_prompt.id,
            model_name="tinyllama",
            temperature=0.7,
            max_tokens=200,
        )

        assert result.response == "Paris is the capital of France."
        assert result.latency_ms == 150.0
        assert result.hallucination_score == 0.05
        assert result.quality_score == 0.9
        assert result.toxicity_score == 0.0
        assert result.token_cost == 0.0005
        assert result.gpu_utilization == 0.45
        assert result.mlflow_run_id == "run_123"

    @patch("app.evaluation.evaluator.OllamaClient")
    @patch("app.evaluation.evaluator.MLflowService")
    async def test_evaluate_prompt_not_found(self, mock_mlflow, mock_ollama_cls, mock_db, sample_model):
        mock_ollama = AsyncMock()
        mock_ollama_cls.return_value = mock_ollama

        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=result)

        evaluator = Evaluator(mock_db)
        with pytest.raises(ValueError, match="not found"):
            await evaluator.evaluate(prompt_id=uuid.uuid4())

        await evaluator.close()

    @patch("app.evaluation.evaluator.OllamaClient")
    @patch("app.evaluation.evaluator.MLflowService")
    @patch("app.evaluation.evaluator.compute_toxicity", return_value={
        "toxicity_score": 0.6, "max_category_score": 0.6,
        "categories": {"profanity": 0.6}, "is_toxic": True,
    })
    async def test_evaluate_toxic_response(
        self, mock_tox, mock_mlflow, mock_ollama_cls, mock_db, sample_prompt, sample_model,
    ):
        mock_ollama = AsyncMock()
        mock_ollama.generate.return_value = {
            "response": "fuck you",
            "latency_ms": 100.0,
            "total_tokens": 10,
            "prompt_tokens": 5,
            "completion_tokens": 5,
        }
        mock_ollama.get_gpu_info.return_value = {"gpu_available": False}
        mock_ollama_cls.return_value = mock_ollama

        mock_mlflow.return_value.start_run = AsyncMock()
        mock_mlflow.return_value.log_evaluation_metrics = MagicMock()
        mock_mlflow.return_value.end_run = MagicMock()
        mock_mlflow.return_value.active_run_id = None

        call_count = 0

        async def execute_side_effect(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = sample_prompt
            else:
                result.scalar_one_or_none.return_value = sample_model
            return result

        mock_db.execute = execute_side_effect
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        evaluator = Evaluator(mock_db)
        result = await evaluator.evaluate(prompt_id=sample_prompt.id)
        assert result.is_toxic is True
        assert result.toxicity_score == 0.6
        await evaluator.close()

    @patch("app.evaluation.evaluator.OllamaClient")
    @patch("app.evaluation.evaluator.MLflowService")
    @patch("app.evaluation.evaluator.compute_hallucination_score", return_value=0.05)
    @patch("app.evaluation.evaluator.compute_quality_score", return_value=0.9)
    @patch("app.evaluation.evaluator.compute_relevance", return_value=0.85)
    @patch("app.evaluation.evaluator.compute_factual_consistency", return_value=0.95)
    @patch("app.evaluation.evaluator.compute_sentence_level_hallucination", return_value=(0.1, []))
    @patch("app.evaluation.evaluator.compute_token_cost", return_value=0.0005)
    @patch("app.evaluation.evaluator.compute_toxicity", return_value={
        "toxicity_score": 0.0, "categories": {}, "is_toxic": False,
    })
    async def test_evaluate_with_llm_model_id(
        self, mock_tox, mock_cost, mock_sentence_hal, mock_factual,
        mock_relevance, mock_quality, mock_hallucination,
        mock_mlflow, mock_ollama_cls, mock_db, sample_prompt, sample_model,
    ):
        mock_ollama = AsyncMock()
        mock_ollama.generate.return_value = {
            "response": "Paris", "latency_ms": 100.0,
            "total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5,
        }
        mock_ollama.get_gpu_info.return_value = {"gpu_available": False}
        mock_ollama_cls.return_value = mock_ollama

        mock_mlflow.return_value.start_run = AsyncMock()
        mock_mlflow.return_value.log_evaluation_metrics = MagicMock()
        mock_mlflow.return_value.end_run = MagicMock()
        mock_mlflow.return_value.active_run_id = "run_456"

        call_count = 0

        async def execute_side_effect(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = sample_prompt
            elif call_count == 2:
                result.scalar_one_or_none.return_value = sample_model
            else:
                result.scalar_one_or_none.return_value = sample_model
            return result

        mock_db.execute = execute_side_effect
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        evaluator = Evaluator(mock_db)
        result = await evaluator.evaluate(
            prompt_id=sample_prompt.id,
            llm_model_id=sample_model.id,
            model_name="test-model",
        )
        assert result.hallucination_score == 0.05
        assert result.mlflow_run_id == "run_456"
        await evaluator.close()
