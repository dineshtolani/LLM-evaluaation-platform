import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from app.evaluation import metrics


@pytest.fixture(autouse=True)
def reset_embedder():
    metrics._embedder = None
    yield
    metrics._embedder = None


class TestQualityScore:
    def test_empty_response(self):
        assert metrics.compute_quality_score("") == 0.0

    def test_very_short(self):
        assert metrics.compute_quality_score("Hi") == 0.0

    def test_good_quality(self):
        text = ("The Eiffel Tower is located in Paris, France. "
                "It was built in 1889 for the World's Fair. "
                "It is one of the most famous landmarks in the world.")
        score = metrics.compute_quality_score(text)
        assert 0.5 <= score <= 1.0

    def test_repeated_words_penalty(self):
        text = "the the the cat in the hat the the"
        score = metrics.compute_quality_score(text)
        assert score <= 0.7

    def test_low_lexical_diversity(self):
        text = "hello hello hello hello hello hello hello hello hello hello hello"
        score = metrics.compute_quality_score(text)
        assert score < 0.6

    def test_three_words_minimum(self):
        assert metrics.compute_quality_score("a b") == 0.0

    def test_scoring_floor(self):
        score = metrics.compute_quality_score("word " * 3)
        assert score >= 0.0


class TestTokenCost:
    def test_basic_cost(self):
        cost = metrics.compute_token_cost(100, 50)
        assert cost == pytest.approx(100 * 0.000003 + 50 * 0.000015)

    def test_zero_tokens(self):
        assert metrics.compute_token_cost(0, 0) == 0.0

    def test_custom_rates(self):
        cost = metrics.compute_token_cost(100, 50, 0.000001, 0.000002)
        assert cost == pytest.approx(100 * 0.000001 + 50 * 0.000002)

    def test_no_cost(self):
        cost = metrics.compute_token_cost(0, 100)
        assert cost > 0


class TestRelevance:
    def _make_cos_sim_return(self, value):
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = value
        return mock_tensor

    @patch("app.evaluation.metrics.SentenceTransformer")
    @patch("app.evaluation.metrics.util.cos_sim")
    def test_basic_relevance(self, mock_cos_sim, mock_st_cls):
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = MagicMock()
        mock_cos_sim.return_value = self._make_cos_sim_return(0.85)

        score = metrics.compute_relevance("What is France?", "France is a country.")
        assert score == 0.85
        mock_model.encode.assert_called()

    @patch("app.evaluation.metrics.SentenceTransformer")
    @patch("app.evaluation.metrics.util.cos_sim")
    def test_empty_prompt(self, mock_cos_sim, mock_st_cls):
        assert metrics.compute_relevance("", "response") == 0.0

    @patch("app.evaluation.metrics.SentenceTransformer")
    @patch("app.evaluation.metrics.util.cos_sim")
    def test_empty_response(self, mock_cos_sim, mock_st_cls):
        assert metrics.compute_relevance("prompt", "") == 0.0


class TestHallucinationScore:
    @patch("app.evaluation.metrics.SentenceTransformer")
    @patch("app.evaluation.metrics.util.cos_sim")
    def test_with_expected_output(self, mock_cos_sim, mock_st_cls):
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 0.9
        mock_cos_sim.return_value = mock_tensor

        score = metrics.compute_hallucination_score(
            "Paris is the capital.",
            "What is the capital of France?",
            "Paris",
        )
        assert score == pytest.approx(0.1, abs=0.01)

    @patch("app.evaluation.metrics.SentenceTransformer")
    @patch("app.evaluation.metrics.util.cos_sim")
    def test_without_expected_output(self, mock_cos_sim, mock_st_cls):
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 0.7
        mock_cos_sim.return_value = mock_tensor

        score = metrics.compute_hallucination_score(
            "Paris is the capital.",
            "What is the capital of France?",
        )
        assert score == pytest.approx(0.3, abs=0.01)

    @patch("app.evaluation.metrics.SentenceTransformer")
    @patch("app.evaluation.metrics.util.cos_sim")
    def test_clamping(self, mock_cos_sim, mock_st_cls):
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = -0.5
        mock_cos_sim.return_value = mock_tensor

        score = metrics.compute_hallucination_score(
            "nonsense", "What is France?", "Paris",
        )
        assert 0.0 <= score <= 1.0


class TestFactualConsistency:
    @patch("app.evaluation.metrics.SentenceTransformer")
    @patch("app.evaluation.metrics.util.cos_sim")
    def test_with_expected_output(self, mock_cos_sim, mock_st_cls):
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 0.85
        mock_cos_sim.return_value = mock_tensor

        score = metrics.compute_factual_consistency("response", "expected")
        assert score == 0.85

    def test_without_expected_output(self):
        assert metrics.compute_factual_consistency("response", None) is None

    def test_empty_response(self):
        assert metrics.compute_factual_consistency("", "expected") is None


class TestSentenceLevelHallucination:
    @patch("app.evaluation.metrics.SentenceTransformer")
    @patch("app.evaluation.metrics.util.cos_sim")
    def test_basic_analysis(self, mock_cos_sim, mock_st_cls):
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 0.8
        mock_cos_sim.return_value = mock_tensor

        worst, results = metrics.compute_sentence_level_hallucination(
            "Paris is great. France is nice.",
            "Paris is the capital of France.",
        )
        assert 0.0 <= worst <= 1.0
        assert len(results) == 2
        assert results[0]["hallucination"] == pytest.approx(0.2, abs=0.01)

    def test_missing_expected(self):
        worst, results = metrics.compute_sentence_level_hallucination("response", "")
        assert worst == 0.0
        assert results == []

    def test_no_valid_sentences(self):
        worst, results = metrics.compute_sentence_level_hallucination("Hi.", "expected")
        assert worst == 0.0
        assert results == []
