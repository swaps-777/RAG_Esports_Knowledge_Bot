from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from rag_evaluator import LLMJudge


def make_judge_with_responses(responses: list[str]) -> LLMJudge:
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=response))]) for response in responses
        ]
        judge = LLMJudge(api_key="test-key")
        judge.client = mock_client
        return judge


def test_grounding_check_returns_correct_score_with_mocked_responses() -> None:
    judge = make_judge_with_responses(["useful_1 supports revenue", "useful_0 unrelated", "useful_0 generic"])

    result = judge.check_grounding("Apple revenue was $85.8B", ["source 1", "source 2", "source 3"])

    assert result["grounding_score"] == pytest.approx(100 / 3)
    assert result["useful_count"] == 1


def test_precision_check_identifies_tp_fp_fn() -> None:
    judge = make_judge_with_responses(
        [
            """
            {
              "claims": [
                {"text": "Revenue was $85.8B", "classification": "TP", "reason": "supported"},
                {"text": "Mac sales declined 3%", "classification": "FP", "reason": "not in sources"}
              ],
              "tp_count": 1,
              "fp_count": 1,
              "fn_count": 0
            }
            """
        ]
    )

    result = judge.check_precision("answer", ["source"])

    assert result["precision_score"] == pytest.approx(50.0)
    assert result["hallucination_count"] == 1
    assert result["has_hallucination"] is True


def test_relevancy_check_identifies_noncommittal_answers() -> None:
    judge = make_judge_with_responses(
        [
            """
            {
              "directly_addresses_query": false,
              "is_noncommittal": true,
              "relevancy_score": 0.2,
              "reason": "vague"
            }
            """
        ]
    )

    result = judge.check_relevancy("What was revenue?", "It depends on many factors.")

    assert result["is_noncommittal"] is True
    assert result["relevancy_score"] == pytest.approx(0.2)