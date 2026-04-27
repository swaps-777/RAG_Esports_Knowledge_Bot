import pytest

from rag_evaluator import AnswerEvaluator


def test_rouge1_with_known_word_overlap() -> None:
    scores = AnswerEvaluator().calculate_rouge(
        generated="Apple announced strong quarterly results",
        reference="Apple reported strong quarterly earnings",
    )

    assert scores["rouge1_f"] == pytest.approx(0.6)


def test_rouge2_with_known_bigram_overlap() -> None:
    scores = AnswerEvaluator().calculate_rouge(
        generated="Apple announced strong quarterly results",
        reference="Apple reported strong quarterly earnings",
    )

    assert scores["rouge2_f"] == pytest.approx(0.25)


def test_fuzzy_ratio_with_identical_strings() -> None:
    scores = AnswerEvaluator().calculate_fuzzy_match("Revenue was $85.8 billion", "Revenue was $85.8 billion")

    assert scores["fuzzy_ratio"] == 100.0


def test_fuzzy_partial_ratio_with_substring_match() -> None:
    scores = AnswerEvaluator().calculate_fuzzy_match(
        generated="In Q3 2024, Apple's revenue was $85.8 billion, up 5% YoY.",
        reference="revenue was $85.8 billion",
    )

    assert scores["fuzzy_partial_ratio"] >= 90.0


def test_semantic_similarity_with_identical_text() -> None:
    score = AnswerEvaluator().calculate_semantic_similarity(
        "Apple reported Q3 revenue",
        "Apple reported Q3 revenue",
    )

    assert score == pytest.approx(1.0)


def test_semantic_similarity_with_completely_different_text() -> None:
    score = AnswerEvaluator().calculate_semantic_similarity(
        "Apple reported Q3 revenue",
        "The gym routine includes squats",
    )

    assert score < 0.2