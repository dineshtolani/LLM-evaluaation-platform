import httpx
import time
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings


class OllamaClient:
    def __init__(self, base_url: str = settings.ollama_base_url):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 40,
        num_ctx: int = 4096,
        use_gpu: bool = True,
    ) -> dict:
        start_time = time.time()

        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "num_ctx": num_ctx,
            },
            "stream": False,
        }

        if system_prompt:
            payload["system"] = system_prompt

        response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()

        latency_ms = (time.time() - start_time) * 1000

        return {
            "response": result.get("response", ""),
            "latency_ms": latency_ms,
            "total_tokens": result.get("eval_count", 0) + result.get("prompt_eval_count", 0),
            "prompt_tokens": result.get("prompt_eval_count", 0),
            "completion_tokens": result.get("eval_count", 0),
            "model": result.get("model", model),
            "raw_response": result,
        }

    async def list_models(self) -> list[dict]:
        response = await self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])

    async def pull_model(self, model_name: str) -> dict:
        response = await self.client.post(
            f"{self.base_url}/api/pull",
            json={"name": model_name, "stream": False},
        )
        response.raise_for_status()
        return response.json()

    async def get_gpu_info(self) -> dict:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                line = result.stdout.strip()
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    return {
                        "gpu_available": True,
                        "gpu_name": parts[3],
                        "gpu_utilization": float(parts[0]) / 100.0,
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                    }
        except Exception:
            pass

        try:
            result = await self.client.get(f"{self.base_url}/api/gpu")
            if result.status_code == 200:
                return result.json()
        except Exception:
            pass

        return {"gpu_available": False, "error": "GPU info not available"}

    async def close(self):
        await self.client.aclose()
