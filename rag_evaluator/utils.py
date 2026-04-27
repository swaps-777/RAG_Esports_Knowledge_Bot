"""Shared utilities for RAG evaluation metrics."""

from __future__ import annotations

import math
import string
from typing import Iterable

import numpy as np

from rag_evaluator.models import RAGEvaluationReport


def tokenize(text: str) -> list[str]:
    """Lowercase text, strip punctuation, and split into whitespace tokens.

    Algorithm:
    1. Validate that the input is not ``None``.
    2. Lowercase text so "Apple" and "apple" match.
    3. Strip punctuation from each token.
    4. Drop empty tokens.

    Formula-style view:
    tokens = [strip_punctuation(word.lower()) for word in text.split()]

    Example:
    ``"Apple, Inc. Revenue!"`` becomes ``["apple", "inc", "revenue"]``.
    """
    if text is None:
        raise ValueError("text must not be None")

    punctuation = str.maketrans("", "", string.punctuation)
    return [word.translate(punctuation) for word in text.lower().split() if word.translate(punctuation)]


def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract contiguous n-grams from a token list.

    Algorithm:
    For each valid start index ``i``, take ``tokens[i:i+n]`` and return it as
    a tuple. If ``n`` is larger than the number of tokens, no n-grams exist.

    Formula:
    ngrams = (tokens[i], ..., tokens[i+n-1]) for i in [0, len(tokens)-n]

    Example:
    tokens ``["apple", "q3", "revenue"]`` with ``n=2`` returns
    ``[("apple", "q3"), ("q3", "revenue")]``.
    """
    if tokens is None:
        raise ValueError("tokens must not be None")
    if n <= 0:
        raise ValueError("n must be greater than 0")

    return [tuple(tokens[index : index + n]) for index in range(max(len(tokens) - n + 1, 0))]


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors using numpy.

    Algorithm:
    Cosine similarity measures the angle between two vectors.

    Formula:
    cosine(A, B) = dot(A, B) / (||A|| * ||B||)

    Example:
    A=[1, 0], B=[1, 0] gives 1/(1*1)=1.0.
    A=[1, 0], B=[0, 1] gives 0/(1*1)=0.0.
    """
    if vec_a is None or vec_b is None:
        raise ValueError("vectors must not be None")
    if len(vec_a) != len(vec_b):
        raise ValueError("vectors must have the same length")
    if not vec_a:
        return 0.0

    arr_a = np.array(vec_a, dtype=float)
    arr_b = np.array(vec_b, dtype=float)
    denominator = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    if math.isclose(denominator, 0.0):
        return 0.0

    return float(np.dot(arr_a, arr_b) / denominator)


def _metric_rows(name: str, metrics: object | None) -> Iterable[tuple[str, str]]:
    if metrics is None:
        yield name, "not run"
        return
    for key, value in metrics.model_dump().items():
        yield f"{name}.{key}", str(value)


def format_report(report: RAGEvaluationReport) -> str:
    """Pretty-print a complete RAG evaluation report as a simple table.

    Algorithm:
    1. Flatten each metric model into ``name=value`` rows.
    2. Calculate the widest metric name for table alignment.
    3. Join rows with newlines for terminal-friendly output.

    Formula-style view:
    row = metric_name.ljust(max_width) + " | " + value

    Example:
    ``SearchMetrics(mrr=1.0, ...)`` appears as ``search.mrr | 1.0``.
    """
    if report is None:
        raise ValueError("report must not be None")

    rows = [("query", report.query), ("timestamp", report.timestamp)]
    rows.extend(_metric_rows("search", report.search_metrics))
    rows.extend(_metric_rows("answer", report.answer_metrics))
    rows.extend(_metric_rows("llm_judge", report.llm_judge_metrics))

    width = max(len(name) for name, _ in rows)
    border = f"+-{'-' * width}-+-{'-' * 24}-+"
    lines = [border, f"| {'Metric'.ljust(width)} | {'Value'.ljust(24)} |", border]
    lines.extend(f"| {name.ljust(width)} | {value[:24].ljust(24)} |" for name, value in rows)
    lines.append(border)
    return "\n".join(lines)