import re
from typing import Optional
from sentence_transformers import SentenceTransformer, util


_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def compute_quality_score(response: str) -> float:
    if not response or len(response.strip()) < 10:
        return 0.0

    score = 0.5

    length = len(response.split())
    if 20 <= length <= 500:
        score += 0.15
    elif length > 500:
        score += 0.1
    else:
        score -= 0.1

    sentences = re.split(r'[.!?]+', response)
    if len(sentences) >= 2:
        score += 0.1

    unique_words = len(set(response.lower().split()))
    total_words = len(response.split())
    if total_words > 0:
        lexical_diversity = unique_words / total_words
        if 0.4 <= lexical_diversity <= 0.9:
            score += 0.15
        elif lexical_diversity > 0.9:
            score += 0.05

    repeated_patterns = re.findall(r'(\b\w+\b)(?:\s+\1\b){2,}', response.lower())
    if repeated_patterns:
        score -= 0.1 * len(repeated_patterns)

    if total_words < 3:
        score = 0.0

    return max(0.0, min(1.0, score))


def compute_relevance(prompt: str, response: str) -> float:
    if not prompt or not response:
        return 0.0

    model = _get_embedder()
    prompt_emb = model.encode(prompt, convert_to_tensor=True)
    response_emb = model.encode(response, convert_to_tensor=True)
    similarity = util.cos_sim(prompt_emb, response_emb).item()
    return max(0.0, min(1.0, float(similarity)))


def compute_hallucination_score(
    response: str,
    prompt: str,
    expected_output: Optional[str] = None,
):
    model = _get_embedder()

    if expected_output:
        response_emb = model.encode(response, convert_to_tensor=True)
        expected_emb = model.encode(expected_output, convert_to_tensor=True)
        consistency = float(util.cos_sim(response_emb, expected_emb).item())
        hallucination = 1.0 - consistency
    else:
        prompt_emb = model.encode(prompt, convert_to_tensor=True)
        response_emb = model.encode(response, convert_to_tensor=True)
        relevance = float(util.cos_sim(prompt_emb, response_emb).item())
        hallucination = 1.0 - relevance

    return max(0.0, min(1.0, float(hallucination)))


def compute_factual_consistency(
    response: str,
    expected_output: Optional[str] = None,
):
    if not expected_output or not response:
        return None

    model = _get_embedder()
    resp_emb = model.encode(response, convert_to_tensor=True)
    expected_emb = model.encode(expected_output, convert_to_tensor=True)
    return float(max(0.0, min(1.0, util.cos_sim(resp_emb, expected_emb).item())))


def compute_sentence_level_hallucination(
    response: str,
    expected_output: str,
) -> tuple[float, list[dict]]:
    if not expected_output or not response:
        return 0.0, []

    model = _get_embedder()
    expected_emb = model.encode(expected_output, convert_to_tensor=True)

    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return 0.0, []

    results = []
    for sentence in sentences:
        sent_emb = model.encode(sentence, convert_to_tensor=True)
        similarity = float(util.cos_sim(sent_emb, expected_emb).item())
        results.append({
            "sentence": sentence,
            "similarity": round(similarity, 4),
            "hallucination": 1.0 - similarity,
        })

    worst_hallucination = max(r["hallucination"] for r in results)

    return worst_hallucination, results


def compute_token_cost(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost_per_token: float = 0.000003,
    completion_cost_per_token: float = 0.000015,
) -> float:
    return (prompt_tokens * prompt_cost_per_token) + (completion_tokens * completion_cost_per_token)
