"""End-to-end RAG evaluator example."""

from __future__ import annotations

import os
from datetime import UTC, datetime

from rag_evaluator import (
    AnswerEvalInput,
    AnswerEvaluator,
    LLMJudge,
    RAGEvaluationReport,
    RetrievedDocument,
    SearchEvaluator,
    SearchResult,
)
from rag_evaluator.utils import format_report


def main() -> None:
    query = "What is Apple's Q3 2024 revenue?"
    source_texts = [
        "Apple Q3 2024 earnings report: revenue was $85.5 Billion, up 5% year-over-year.",
        "Apple stock price today moved during regular trading.",
        "Apple Q2 2024 earnings report covered prior-quarter revenue.",
        "Microsoft Q3 2024 report highlighted cloud growth.",
        "Apple Revenue history 2020-24 includes Q3 2024 revenue of $85.5 Billion.",
    ]
    reference_answer = "Apple reported Q3 2024 revenue of $85.5 Billion, up 5% year-over-year."
    generated_answer = (
        "Apple's Q3 2024 revenue was 85.5 billion, representing a 5% increase "
        "compared to the same quarter last year."
    )

    search_result = SearchResult(
        query=query,
        total_relevant_in_db=2,
        retrieved_docs=[
            RetrievedDocument(doc_id="apple-q3", content=source_texts[0], relevance_score=1.0, is_relevant=True),
            RetrievedDocument(doc_id="stock", content=source_texts[1], relevance_score=0.0, is_relevant=False),
            RetrievedDocument(doc_id="apple-q2", content=source_texts[2], relevance_score=0.4, is_relevant=False),
            RetrievedDocument(doc_id="msft-q3", content=source_texts[3], relevance_score=0.0, is_relevant=False),
            RetrievedDocument(doc_id="apple-history", content=source_texts[4], relevance_score=0.8, is_relevant=True),
        ],
    )

    search_metrics = SearchEvaluator().evaluate([search_result])
    answer_metrics = AnswerEvaluator().evaluate(
        AnswerEvalInput(
            query=query,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
            source_documents=source_texts,
        )
    )

    llm_judge_metrics = None
    if os.getenv("OPENAI_API_KEY"):
        judge = LLMJudge()
        llm_judge_metrics = judge.evaluate(query, generated_answer, source_texts)

    report = RAGEvaluationReport(
        query=query,
        search_metrics=search_metrics,
        answer_metrics=answer_metrics,
        llm_judge_metrics=llm_judge_metrics,
        timestamp=datetime.now(UTC).isoformat(),
    )
    print(format_report(report))


if __name__ == "__main__":
    main()