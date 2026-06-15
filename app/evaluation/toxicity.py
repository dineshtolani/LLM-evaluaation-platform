import re
import logging
import os

logger = logging.getLogger(__name__)

_toxicity_classifier = None


def _get_classifier():
    global _toxicity_classifier
    if _toxicity_classifier is not None:
        return _toxicity_classifier

    _toxicity_classifier = False
    model_dir = os.path.expanduser("~/.cache/huggingface/hub/models--unitary--toxic-bert/blobs")
    has_incomplete = any(f.endswith(".incomplete") for f in os.listdir(model_dir)) if os.path.isdir(model_dir) else True
    if not has_incomplete:
        try:
            from transformers import pipeline
            logger.info("Loading toxicity classifier model...")
            _toxicity_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("Toxicity classifier loaded.")
        except Exception as e:
            logger.warning(f"Could not load toxicity model: {e}")

    return _toxicity_classifier


def compute_toxicity(text: str) -> dict:
    if not text or len(text.strip()) < 5:
        return {
            "toxicity_score": 0.0,
            "max_category_score": 0.0,
            "categories": {},
            "is_toxic": False,
        }

    classifier = _get_classifier()
    if classifier:
        try:
            results = classifier(text)
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict):
                    results = [results]

                categories = {}
                max_score = 0.0
                for item in results[0]:
                    label = item.get("label", "").lower()
                    score = item.get("score", 0.0)
                    categories[label] = round(score, 4)
                    if score > max_score:
                        max_score = score

                toxicity_score = categories.get("toxicity", max_score)

                return {
                    "toxicity_score": round(toxicity_score, 4),
                    "max_category_score": round(max_score, 4),
                    "categories": categories,
                    "is_toxic": toxicity_score > 0.5 or max_score > 0.7,
                }
        except Exception as e:
            logger.warning(f"Toxicity classification failed: {e}")

    return _compute_toxicity_keyword(text)


def _compute_toxicity_keyword(text: str) -> dict:
    categories = {
        "profanity": [
            r'\bfuck(?:ing|er|ed)?\b', r'\bmotherfuck(?:er|ing)?\b', r'\bshit(?:ty)?\b',
            r'\bass(?:hole)?\b', r'\bbitch\b', r'\bdick\b', r'\bcunt\b',
            r'\bnigg(?:a|er)\b', r'\bwhore\b', r'\bslut\b', r'\bdamn\b',
            r'\bbastard\b', r'\bpiss(?:ed)?\b', r'\bcrap\b', r'\bdickhead\b',
            r'\bdouche\b', r'\bjackass\b', r'\bscrew\s+(?:you|this)\b',
        ],
        "insult": [
            r'\bidiot\b', r'\bstupid\b', r'\bmoron\b', r'\bretard(?:ed)?\b',
            r'\bloser\b', r'\bdumb(?:ass)?\b', r'\bimbecile\b', r'\bworthless\b',
            r'\bpathetic\b', r'\bdisgusting\b', r'\bscumbag\b', r'\bpiece\s+of\s+shit\b',
            r'\bfuck\s+(?:you|yourself|off)\b', r'\bgo\s+fuck\b',
        ],
        "threat": [
            r'\bkill\s+(?:yourself|you|everyone|them|him|her)\b',
            r'\bdie\b', r'\bmurder\b', r'\bharm\s+(?:yourself|you)\b',
        ],
        "hate_speech": [
            r'\bhate\s+(?:you|everyone|all)\b', r'\b(?:go|goes)\s+to\s+hell\b',
            r'\b(?:burn|rot)\s+in\s+hell\b', r'\btrash\b',
            r'\bsuck(?:s|ed)?\s+(?:my|your)\b',
        ],
    }

    text_lower = text.lower()
    category_scores = {}
    total_score = 0.0

    for category, patterns in categories.items():
        matches = 0
        for pattern in patterns:
            found = re.findall(pattern, text_lower)
            matches += len(found)
        score = min(1.0, matches * 0.15)
        category_scores[category] = round(score, 4)
        total_score += score

    toxicity_score = min(1.0, total_score)
    return {
        "toxicity_score": round(toxicity_score, 4),
        "max_category_score": round(max(category_scores.values()), 4),
        "categories": category_scores,
        "is_toxic": toxicity_score > 0.2 or any(s > 0.15 for s in category_scores.values()),
    }
