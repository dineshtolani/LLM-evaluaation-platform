import pytest
from app.evaluation.toxicity import compute_toxicity


class TestToxicityKeywordDetection:
    def test_clean_text(self):
        r = compute_toxicity("What a beautiful day!")
        assert r["toxicity_score"] == 0.0
        assert r["is_toxic"] is False

    def test_empty_text(self):
        r = compute_toxicity("")
        assert r["toxicity_score"] == 0.0

    def test_short_text(self):
        r = compute_toxicity("hi")
        assert r["toxicity_score"] == 0.0

    def test_single_profanity(self):
        r = compute_toxicity("this is fucking stupid")
        assert r["toxicity_score"] > 0
        assert r["categories"]["profanity"] > 0
        assert r["is_toxic"] is True

    def test_multiple_profanities(self):
        r = compute_toxicity("fuck this shit")
        score = r["toxicity_score"]
        assert score == 0.30

    def test_insult_detection(self):
        r = compute_toxicity("you are an idiot")
        assert r["categories"]["insult"] == 0.15
        assert r["is_toxic"] is False

    def test_directed_insult(self):
        r = compute_toxicity("go and fuck yourself")
        assert r["categories"]["profanity"] > 0
        assert r["categories"]["insult"] > 0
        assert r["toxicity_score"] == 0.30

    def test_fuck_you(self):
        r = compute_toxicity("fuck you")
        assert r["toxicity_score"] == 0.30
        assert r["is_toxic"] is True

    def test_threat_detection(self):
        r = compute_toxicity("I will kill you")
        assert r["categories"]["threat"] > 0

    def test_hate_speech(self):
        r = compute_toxicity("go to hell")
        assert r["categories"]["hate_speech"] > 0

    def test_mixed_categories(self):
        r = compute_toxicity("you are a fucking worthless idiot, go kill yourself")
        assert r["toxicity_score"] > 0.30
        assert r["is_toxic"] is True

    def test_motherfucker(self):
        r = compute_toxicity("you motherfucker")
        assert r["categories"]["profanity"] > 0
        assert r["is_toxic"] is False

    def test_piece_of_shit(self):
        r = compute_toxicity("you piece of shit")
        assert r["categories"]["insult"] > 0
        assert r["is_toxic"] is True

    def test_case_insensitivity(self):
        r1 = compute_toxicity("FUCK YOU")
        r2 = compute_toxicity("fuck you")
        assert r1["toxicity_score"] == r2["toxicity_score"]

    def test_special_characters(self):
        r = compute_toxicity("fuck!!! you!!!")
        assert r["categories"]["profanity"] > 0

    def test_non_toxic_phrases(self):
        phrases = [
            "I love programming",
            "The weather is nice today",
            "Can you help me with this?",
            "This is a great project",
            "Thank you for your help",
            "I appreciate your work",
        ]
        for phrase in phrases:
            r = compute_toxicity(phrase)
            assert r["toxicity_score"] == 0.0, f"False positive: {phrase}"


class TestToxicityThresholds:
    def test_above_default_threshold(self):
        r = compute_toxicity("you are a fucking idiot")
        assert r["is_toxic"] is True

    def test_single_word_below_threshold(self):
        r = compute_toxicity("you are an idiot")
        assert r["is_toxic"] is False

    def test_borderline_profanity(self):
        r = compute_toxicity("this is crap")
        assert r["toxicity_score"] == 0.15
        assert r["is_toxic"] is False
