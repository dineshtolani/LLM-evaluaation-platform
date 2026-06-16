from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

evaluations_total = Counter(
    "llm_evaluations_total",
    "Total evaluations processed",
    ["status", "model_name"],
)

evaluations_latency = Histogram(
    "llm_evaluation_latency_seconds",
    "Evaluation latency in seconds",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

hallucination_score = Histogram(
    "llm_hallucination_score",
    "Hallucination score distribution",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
)

toxicity_score = Histogram(
    "llm_toxicity_score",
    "Toxicity score distribution",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
)

quality_score = Histogram(
    "llm_quality_score",
    "Quality score distribution",
    buckets=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

token_usage = Counter(
    "llm_token_usage_total",
    "Total tokens consumed",
    ["type"],
)

gpu_utilization = Gauge(
    "llm_gpu_utilization_ratio",
    "GPU utilization ratio (0-1)",
)

gpu_memory_mb = Gauge(
    "llm_gpu_memory_used_mb",
    "GPU memory used in MB",
)

toxic_responses = Counter(
    "llm_toxic_responses_total",
    "Total toxic responses detected",
)

ollama_requests = Counter(
    "llm_ollama_requests_total",
    "Total Ollama API requests",
    ["status"],
)

alert_triggered = Counter(
    "llm_alerts_triggered_total",
    "Total alerts triggered",
    ["metric"],
)


def track_evaluation(metric_func):
    @wraps(metric_func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await metric_func(*args, **kwargs)
            elapsed = time.time() - start
            model_name = kwargs.get("model_name", "unknown")
            evaluations_total.labels(status="success", model_name=model_name).inc()
            evaluations_latency.observe(elapsed)
            if hasattr(result, "hallucination_score") and result.hallucination_score is not None:
                hallucination_score.observe(result.hallucination_score)
            if hasattr(result, "toxicity_score") and result.toxicity_score is not None:
                toxicity_score.observe(result.toxicity_score)
            if hasattr(result, "quality_score") and result.quality_score is not None:
                quality_score.observe(result.quality_score)
            if hasattr(result, "total_tokens"):
                token_usage.labels(type="prompt").inc(result.prompt_tokens or 0)
                token_usage.labels(type="completion").inc(result.completion_tokens or 0)
            if hasattr(result, "is_toxic") and result.is_toxic:
                toxic_responses.inc()
            return result
        except Exception:
            model_name = kwargs.get("model_name", "unknown")
            evaluations_total.labels(status="error", model_name=model_name).inc()
            raise
    return wrapper
