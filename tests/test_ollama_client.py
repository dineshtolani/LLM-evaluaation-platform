import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.evaluation.ollama_client import OllamaClient

pytestmark = pytest.mark.asyncio(loop_scope="function")


@pytest.fixture
def mock_httpx():
    with patch("app.evaluation.ollama_client.httpx.AsyncClient") as m:
        client = AsyncMock()
        m.return_value = client
        yield client


class TestOllamaGenerate:
    async def test_basic_generate(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Paris is the capital of France.",
            "eval_count": 10,
            "prompt_eval_count": 5,
            "model": "tinyllama",
        }
        mock_httpx.post = AsyncMock(return_value=mock_response)

        client = OllamaClient(base_url="http://test:11434")
        result = await client.generate(model="tinyllama", prompt="What is France?")

        assert result["response"] == "Paris is the capital of France."
        assert result["completion_tokens"] == 10
        assert result["prompt_tokens"] == 5
        assert result["total_tokens"] == 15
        assert "latency_ms" in result
        assert result["model"] == "tinyllama"
        await client.close()

    async def test_generate_with_system_prompt(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Be helpful.", "eval_count": 3, "prompt_eval_count": 2,
        }
        mock_httpx.post = AsyncMock(return_value=mock_response)

        client = OllamaClient(base_url="http://test:11434")
        result = await client.generate(
            model="tinyllama", prompt="Hi", system_prompt="Be nice",
        )
        assert result["response"] == "Be helpful."
        await client.close()

    async def test_generate_empty_response(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_httpx.post = AsyncMock(return_value=mock_response)

        client = OllamaClient(base_url="http://test:11434")
        result = await client.generate(model="tinyllama", prompt="Hi")
        assert result["response"] == ""
        assert result["total_tokens"] == 0
        await client.close()


class TestOllamaListModels:
    async def test_list_models(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "tinyllama"}, {"name": "llama3.2"}],
        }
        mock_httpx.get = AsyncMock(return_value=mock_response)

        client = OllamaClient(base_url="http://test:11434")
        models = await client.list_models()
        assert len(models) == 2
        assert models[0]["name"] == "tinyllama"
        await client.close()

    async def test_list_models_empty(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_httpx.get = AsyncMock(return_value=mock_response)

        client = OllamaClient(base_url="http://test:11434")
        models = await client.list_models()
        assert models == []
        await client.close()


class TestOllamaPullModel:
    async def test_pull_model(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_httpx.post = AsyncMock(return_value=mock_response)

        client = OllamaClient(base_url="http://test:11434")
        result = await client.pull_model("tinyllama")
        assert result["status"] == "success"
        await client.close()


class TestGPUInfo:
    async def test_gpu_info_via_nvidia_smi(self, mock_httpx):
        with patch("subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "45, 2048, 4096, RTX A500"
            mock_run.return_value = mock_proc

            client = OllamaClient()
            info = await client.get_gpu_info()
            assert info["gpu_available"] is True
            assert info["gpu_name"] == "RTX A500"
            assert info["gpu_utilization"] == 0.45
            assert info["memory_used_mb"] == 2048.0
            await client.close()

    async def test_gpu_info_fallback_to_api(self, mock_httpx):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"gpu_available": True, "name": "RTX"}
            mock_httpx.get = AsyncMock(return_value=mock_response)

            client = OllamaClient(base_url="http://test:11434")
            info = await client.get_gpu_info()
            assert info["gpu_available"] is True
            await client.close()

    async def test_gpu_info_unavailable(self, mock_httpx):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("no gpu")
            mock_httpx.get = AsyncMock(side_effect=Exception("api failed"))

            client = OllamaClient(base_url="http://test:11434")
            info = await client.get_gpu_info()
            assert info["gpu_available"] is False
            await client.close()
