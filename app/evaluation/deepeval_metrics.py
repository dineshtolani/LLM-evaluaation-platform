import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DeepEvalResults:
    faithfulness_score: Optional[float] = None
    faithfulness_reason: Optional[str] = None
    hallucination_score: Optional[float] = None
    hallucination_reason: Optional[str] = None
    toxicity_score: Optional[float] = None
    toxicity_reason: Optional[str] = None
    bias_score: Optional[float] = None
    bias_reason: Optional[str] = None
    g_eval_score: Optional[float] = None
    g_eval_reason: Optional[str] = None


DEEPEVAL_AVAILABLE = False
try:
    from deepeval.models import OllamaModel
    from deepeval.metrics import (
        FaithfulnessMetric,
        HallucinationMetric,
        ToxicityMetric,
        BiasMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import BaseMetric

    DEEPEVAL_AVAILABLE = True
except ImportError:
    OllamaModel = None
    LLMTestCase = None
    FaithfulnessMetric = None
    HallucinationMetric = None
    ToxicityMetric = None
    BiasMetric = None
    GEval = None
    BaseMetric = None


def run_deepeval_metrics(
    input_text: str,
    actual_output: str,
    expected_output: Optional[str] = None,
    context: Optional[list[str]] = None,
    ollama_base_url: str = "http://localhost:11434",
    model_name: str = "tinyllama",
    use_g_eval: bool = False,
) -> DeepEvalResults:
    if not DEEPEVAL_AVAILABLE:
        logger.warning("deepeval not installed, skipping metrics")
        return DeepEvalResults()

    if not expected_output and not context:
        logger.info("No expected_output or context provided, skipping deepeval metrics")
        return DeepEvalResults()

    try:
        model = OllamaModel(model=model_name, base_url=ollama_base_url)
        _ = model.get_model_name()
        model.load_model()
    except Exception as e:
        logger.warning(f"Failed to load OllamaModel for deepeval: {e}")
        return DeepEvalResults()

    ctx = context or [input_text]
    tc = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output or "",
        context=ctx,
        retrieval_context=ctx,
    )

    results = DeepEvalResults()
    metrics_to_run = []

    if FaithfulnessMetric is not None:
        metrics_to_run.append(("faithfulness", FaithfulnessMetric(model=model)))
    if HallucinationMetric is not None:
        metrics_to_run.append(("hallucination", HallucinationMetric(model=model)))
    if ToxicityMetric is not None:
        metrics_to_run.append(("toxicity", ToxicityMetric(model=model)))
    if BiasMetric is not None:
        metrics_to_run.append(("bias", BiasMetric(model=model)))
    if use_g_eval and GEval is not None:
        metrics_to_run.append(
            ("g_eval", GEval(model=model, criteria="Overall quality and relevance"))
        )

    for name, metric in metrics_to_run:
        try:
            metric.measure(tc)
            score = metric.score
            reason = getattr(metric, "reason", "")
            setattr(results, f"{name}_score", score)
            setattr(results, f"{name}_reason", reason)
            logger.info(
                "DeepEval %s: %.4f — %s",
                name,
                score,
                reason[:100] if reason else "",
            )
        except Exception as e:
            logger.warning("DeepEval %s failed: %s", name, e)

    return results
