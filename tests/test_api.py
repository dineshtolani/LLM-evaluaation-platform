import uuid
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

import pytest
from httpx import AsyncClient
from app.models.prompt import Prompt
from app.models.evaluation import Evaluation
from app.evaluation.evaluator import Evaluator

pytestmark = pytest.mark.asyncio(loop_scope="function")


def make_mock_evaluation(prompt_id, model_id, **overrides):
    """Helper to build a fake Evaluation object returned by mocked Evaluator."""
    now = datetime.now(timezone.utc)
    data = dict(
        id=uuid.uuid4(),
        prompt_id=prompt_id,
        llm_model_id=model_id,
        response="Paris is the capital of France.",
        latency_ms=120.5,
        total_tokens=42,
        prompt_tokens=10,
        completion_tokens=32,
        token_cost=0.00051,
        hallucination_score=0.05,
        nli_hallucination_score=None,
        toxicity_score=0.0,
        toxicity_categories_json='{"profanity":0.0,"insult":0.0,"threat":0.0,"hate_speech":0.0}',
        is_toxic=False,
        quality_score=0.92,
        relevance_score=0.88,
        factual_consistency=0.95,
        sentence_analysis_json=None,
        gpu_utilization=None,
        gpu_memory_used_mb=None,
        mlflow_run_id=None,
        params_json='{"temperature":0.7}',
        error_message=None,
        created_at=now,
    )
    data.update(overrides)
    return Evaluation(**data)


class TestPrompts:
    async def test_create_prompt(self, client: AsyncClient):
        resp = await client.post("/api/prompts", json={
            "name": "my_q",
            "content": "What is Python?",
            "category": "qa",
            "tags": "programming",
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "my_q"
        assert body["category"] == "qa"
        assert body["version"] == 1

    async def test_list_prompts(self, client: AsyncClient, sample_prompt: Prompt):
        resp = await client.get("/api/prompts")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        assert any(p["name"] == "test_prompt" for p in body["items"])

    async def test_get_prompt(self, client: AsyncClient, sample_prompt: Prompt):
        resp = await client.get(f"/api/prompts/{sample_prompt.id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "test_prompt"

    async def test_get_prompt_not_found(self, client: AsyncClient):
        resp = await client.get(f"/api/prompts/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_update_prompt(self, client: AsyncClient, sample_prompt: Prompt):
        resp = await client.put(f"/api/prompts/{sample_prompt.id}", json={
            "name": "renamed",
            "content": "Updated content",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "renamed"
        assert body["version"] == 2

    async def test_delete_prompt(self, client: AsyncClient, sample_prompt: Prompt):
        resp = await client.delete(f"/api/prompts/{sample_prompt.id}")
        assert resp.status_code == 204

        resp = await client.get(f"/api/prompts/{sample_prompt.id}")
        assert resp.status_code == 404


class TestModels:
    async def test_register_model(self, client: AsyncClient):
        resp = await client.post("/api/models", json={
            "name": "llama3.2",
            "provider": "ollama",
            "model_type": "open-source",
        })
        assert resp.status_code == 201
        assert resp.json()["name"] == "llama3.2"

    async def test_list_models(self, client: AsyncClient, sample_model):
        resp = await client.get("/api/models")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    async def test_get_model(self, client: AsyncClient, sample_model):
        resp = await client.get(f"/api/models/{sample_model.id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "test-model"

    async def test_get_model_not_found(self, client: AsyncClient):
        resp = await client.get(f"/api/models/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_delete_model(self, client: AsyncClient, sample_model):
        resp = await client.delete(f"/api/models/{sample_model.id}")
        assert resp.status_code == 204


class TestAlerts:
    async def test_create_alert(self, client: AsyncClient):
        resp = await client.post("/api/alerts", json={
            "name": "high-latency",
            "metric": "latency",
            "operator": "gt",
            "threshold": 5000.0,
        })
        assert resp.status_code == 201
        assert resp.json()["name"] == "high-latency"

    async def test_list_alerts(self, client: AsyncClient, sample_alert):
        resp = await client.get("/api/alerts")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    async def test_get_alert(self, client: AsyncClient, sample_alert):
        resp = await client.get(f"/api/alerts/{sample_alert.id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "test-alert"

    async def test_get_alert_not_found(self, client: AsyncClient):
        resp = await client.get(f"/api/alerts/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_update_alert(self, client: AsyncClient, sample_alert):
        resp = await client.put(f"/api/alerts/{sample_alert.id}", json={
            "threshold": 9999.0,
        })
        assert resp.status_code == 200
        assert resp.json()["threshold"] == 9999.0

    async def test_delete_alert(self, client: AsyncClient, sample_alert):
        resp = await client.delete(f"/api/alerts/{sample_alert.id}")
        assert resp.status_code == 204


class TestEvaluations:
    async def test_create_evaluation(
        self, client: AsyncClient, sample_prompt: Prompt, sample_model,
    ):
        mock_eval = make_mock_evaluation(sample_prompt.id, sample_model.id)

        with patch.object(Evaluator, "evaluate", new_callable=AsyncMock, return_value=mock_eval):
            resp = await client.post("/api/evaluations", json={
                "prompt_id": str(sample_prompt.id),
                "model_name": "test-model",
            })

        assert resp.status_code == 201
        body = resp.json()
        assert body["response"] == "Paris is the capital of France."
        assert body["hallucination_score"] == 0.05

    async def test_list_evaluations(
        self, client: AsyncClient, db_session, sample_prompt: Prompt, sample_model,
    ):
        mock_eval = make_mock_evaluation(sample_prompt.id, sample_model.id)
        db_session.add(mock_eval)
        await db_session.commit()

        resp = await client.get("/api/evaluations")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1

    async def test_evaluation_stats(
        self, client: AsyncClient, db_session, sample_prompt: Prompt, sample_model,
    ):
        for i in range(3):
            e = make_mock_evaluation(sample_prompt.id, sample_model.id,
                                     latency_ms=100.0 + i * 50,
                                     total_tokens=30 + i * 10)
            db_session.add(e)
        await db_session.commit()

        resp = await client.get("/api/evaluations/stats")
        assert resp.status_code == 200
        s = resp.json()
        assert s["total_evaluations"] == 3
        assert s["avg_latency_ms"] == 150.0

    async def test_batch_evaluate(
        self, client: AsyncClient, sample_prompt: Prompt, sample_model,
    ):
        mock_eval = make_mock_evaluation(sample_prompt.id, sample_model.id)

        with patch.object(Evaluator, "evaluate", new_callable=AsyncMock, return_value=mock_eval):
            resp = await client.post("/api/evaluations/batch", json={
                "prompt_ids": [str(sample_prompt.id)],
                "model_name": "test-model",
            })

        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["response"] == mock_eval.response


class TestToxicity:
    async def test_detox_clean(self, client: AsyncClient):
        resp = await client.post("/api/evaluations/detox", json={
            "text": "What a beautiful day!",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["toxicity_score"] == 0.0
        assert body["is_toxic"] is False
        assert body["is_blocked"] is False

    async def test_detox_profane(self, client: AsyncClient):
        resp = await client.post("/api/evaluations/detox", json={
            "text": "This is fucking terrible shit",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["toxicity_score"] > 0
        assert body["toxicity_score"] >= 0.30

    async def test_detox_block_threshold(self, client: AsyncClient):
        resp = await client.post("/api/evaluations/detox", json={
            "text": "fuck you",
            "threshold": 0.1,
        })
        assert resp.status_code == 200
        assert resp.json()["is_blocked"] is True


class TestToxicityReport:
    async def test_empty_report(self, client: AsyncClient):
        resp = await client.get("/api/evaluations/toxicity-report")
        assert resp.status_code == 200
        assert resp.json()["total_evaluations"] == 0

    async def test_report_with_data(
        self, client: AsyncClient, db_session, sample_prompt: Prompt, sample_model,
    ):
        e = make_mock_evaluation(
            sample_prompt.id, sample_model.id,
            toxicity_score=0.45,
            toxicity_categories_json='{"profanity":0.45,"insult":0.0,"threat":0.0,"hate_speech":0.0}',
            is_toxic=True,
        )
        db_session.add(e)
        await db_session.commit()

        resp = await client.get("/api/evaluations/toxicity-report")
        assert resp.status_code == 200
        r = resp.json()
        assert r["total_evaluations"] == 1
        assert r["toxic_count"] == 1
        assert r["toxic_percentage"] == 100.0
        assert len(r["most_toxic"]) >= 1


class TestHallucinationReport:
    async def test_empty_report(self, client: AsyncClient):
        resp = await client.get("/api/evaluations/hallucination-report")
        assert resp.status_code == 200
        assert resp.json()["total_evaluations"] == 0

    async def test_report_with_data(
        self, client: AsyncClient, db_session, sample_prompt: Prompt, sample_model,
    ):
        e = make_mock_evaluation(
            sample_prompt.id, sample_model.id,
            hallucination_score=0.85, quality_score=0.3,
        )
        db_session.add(e)
        await db_session.commit()

        resp = await client.get("/api/evaluations/hallucination-report")
        assert resp.status_code == 200
        r = resp.json()
        assert r["total_evaluations"] == 1
        assert r["avg_hallucination_score"] == 0.85
        assert r["high_hallucination_count"] == 1
        assert len(r["most_hallucinated"]) >= 1


class TestRoot:
    async def test_root(self, client: AsyncClient):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert resp.json()["app"] == "LLM Eval Platform"

    async def test_health(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
