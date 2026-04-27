import pytest

from rag_evaluator import RetrievedDocument, SearchEvaluator, SearchResult


def make_result(relevance_flags: list[bool], total_relevant: int | None = None) -> SearchResult:
    return SearchResult(
        query="test query",
        total_relevant_in_db=sum(relevance_flags) if total_relevant is None else total_relevant,
        retrieved_docs=[
            RetrievedDocument(
                doc_id=f"doc-{index}",
                content=f"content {index}",
                relevance_score=1.0 if is_relevant else 0.0,
                is_relevant=is_relevant,
            )
            for index, is_relevant in enumerate(relevance_flags)
        ],
    )


def test_mrr_with_first_result_relevant() -> None:
    assert SearchEvaluator().calculate_mrr([make_result([True, False, False])]) == 1.0


def test_mrr_with_first_relevant_at_position_three() -> None:
    assert SearchEvaluator().calculate_mrr([make_result([False, False, True])]) == pytest.approx(1 / 3)


def test_mrr_with_no_relevant_results() -> None:
    assert SearchEvaluator().calculate_mrr([make_result([False, False, False], total_relevant=1)]) == 0.0


def test_precision_recall_with_known_counts() -> None:
    precision, recall, f1 = SearchEvaluator().calculate_precision_recall_f1(
        make_result([True, True, True, False, False], total_relevant=10)
    )

    assert precision == pytest.approx(0.6)
    assert recall == pytest.approx(0.3)
    assert f1 == pytest.approx(0.4)


def test_map_with_notes_example() -> None:
    result = make_result([False, True, False, True, True, False, False, True, False, False])

    assert SearchEvaluator().calculate_map([result]) == pytest.approx(0.525)


def test_ndcg_with_binary_relevance_example() -> None:
    result = make_result([False, True, False, True, True])

    assert SearchEvaluator().calculate_ndcg(result) == pytest.approx(0.68, abs=0.01)


def test_f1_balance_penalizes_imbalanced_precision_and_recall() -> None:
    evaluator = SearchEvaluator()
    _, _, system_a_f1 = evaluator.calculate_precision_recall_f1(make_result([True] * 9 + [False], total_relevant=90))
    _, _, system_b_f1 = evaluator.calculate_precision_recall_f1(make_result([True] * 6 + [False] * 4, total_relevant=10))

    assert system_a_f1 == pytest.approx(0.18)
    assert system_b_f1 == pytest.approx(0.60)
    assert system_b_f1 > system_a_f1