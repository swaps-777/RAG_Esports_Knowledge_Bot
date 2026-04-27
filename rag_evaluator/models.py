"""Pydantic models for RAG evaluation inputs, metrics, and reports."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """A document returned by retrieval for one query."""

    doc_id: str
    content: str
    relevance_score: float = 0.0
    is_relevant: bool = False


class SearchResult(BaseModel):
    """All retrieved documents for a query plus relevance ground truth."""

    query: str
    retrieved_docs: list[RetrievedDocument]
    total_relevant_in_db: int = Field(ge=0)


class AnswerEvalInput(BaseModel):
    """Generated answer, reference answer, and source chunks for evaluation."""

    query: str
    generated_answer: str
    reference_answer: str
    source_documents: list[str]


class SearchMetrics(BaseModel):
    """Aggregate retrieval quality metrics."""

    mrr: float
    map_score: float
    precision: float
    recall: float
    f1: float
    ndcg: float


class AnswerMetrics(BaseModel):
    """Generated-answer quality metrics."""

    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    fuzzy_ratio: float
    fuzzy_partial_ratio: float
    semantic_similarity: float | None = None


class LLMJudgeMetrics(BaseModel):
    """LLM-as-a-judge quality and hallucination metrics."""

    grounding_score: float
    precision_score: float
    hallucination_count: int
    relevancy_score: float
    is_noncommittal: bool
    details: dict


class RAGEvaluationReport(BaseModel):
    """Complete evaluation report for a single RAG query."""

    query: str
    search_metrics: SearchMetrics | None = None
    answer_metrics: AnswerMetrics | None = None
    llm_judge_metrics: LLMJudgeMetrics | None = None
    timestamp: str