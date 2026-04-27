"""Search and retrieval metrics for RAG systems."""

from __future__ import annotations

import math

from rag_evaluator.models import SearchMetrics, SearchResult


class SearchEvaluator:
    """Evaluate whether retrieval returned the right documents in the right order."""

    def calculate_mrr(self, results: list[SearchResult]) -> float:
        """Calculate Mean Reciprocal Rank (MRR).

        Algorithm:
        MRR answers: how quickly do we find the first relevant document?
        For each query, find the 1-indexed rank of the first relevant document.
        The reciprocal rank for that query is ``1 / rank``. If no relevant
        document is found, that query contributes ``0.0``. Final MRR is the mean
        of all query reciprocal ranks.

        Formula:
        MRR = (1 / Q) * sum(1 / rank_i)
        where ``rank_i`` is the first relevant position for query ``i``.

        Worked example:
        Retrieved: [irrelevant, RELEVANT, irrelevant, relevant]
        First relevant at position 2, so reciprocal rank = 1/2 = 0.5.

        Range: 0 to 1. Good value: > 0.8.
        """
        if results is None:
            raise ValueError("results must not be None")
        if not results:
            return 0.0

        reciprocal_ranks: list[float] = []
        for result in results:
            first_relevant_rank = 0
            for index, document in enumerate(result.retrieved_docs, start=1):
                if document.is_relevant:
                    first_relevant_rank = index
                    break

            # Query MRR contribution = 1 / first relevant rank, or 0 if absent.
            reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

        # Mean across queries = sum(query reciprocal ranks) / query count.
        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    def calculate_precision_recall_f1(self, result: SearchResult) -> tuple[float, float, float]:
        """Calculate precision, recall, and F1 for one query result.

        Algorithm:
        Precision asks: of what we retrieved, how many are actually relevant?
        Recall asks: of all relevant docs that exist, how many did we find?
        F1 is the harmonic mean. It penalizes imbalance because both precision
        and recall must be strong for F1 to be high.

        Formulas:
        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / total_relevant_in_db
        F1 = 2 * (precision * recall) / (precision + recall)

        Worked example:
        DB has 10 relevant docs about "Apple Q3 2024".
        Retrieved 5 docs: 3 relevant, 2 not.
        Precision = 3/5 = 0.6.
        Recall = 3/10 = 0.3.
        F1 = 2*(0.6*0.3)/(0.6+0.3) = 0.4.

        Ranges: 0 to 1. Good precision > 0.8, recall > 0.7, F1 > 0.7.
        """
        if result is None:
            raise ValueError("result must not be None")

        total_retrieved = len(result.retrieved_docs)
        relevant_retrieved = sum(1 for document in result.retrieved_docs if document.is_relevant)

        # Precision = relevant retrieved / all retrieved.
        precision = relevant_retrieved / total_retrieved if total_retrieved else 0.0
        # Recall = relevant retrieved / all relevant documents known to exist.
        recall = relevant_retrieved / result.total_relevant_in_db if result.total_relevant_in_db else 0.0
        # F1 harmonic mean; avoid division by zero when both precision and recall are 0.
        f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0

        return precision, recall, f1

    def calculate_map(self, results: list[SearchResult]) -> float:
        """Calculate Mean Average Precision (MAP).

        Algorithm:
        MAP answers: across all relevant documents, how good is our ranking?
        For each query, scan results in order. Every time a relevant document is
        found, calculate precision at that position. Average those precision
        values to get Average Precision (AP). MAP is the mean AP across queries.

        Formula:
        AP = sum(precision@k for each relevant doc at rank k) / relevant_retrieved
        MAP = mean(AP for each query)

        Worked example:
        Results: [X, yes, X, yes, yes, X, X, yes, X, X]
        At pos 2: precision = 1/2 = 0.50
        At pos 4: precision = 2/4 = 0.50
        At pos 5: precision = 3/5 = 0.60
        At pos 8: precision = 4/8 = 0.50
        AP = (0.50 + 0.50 + 0.60 + 0.50) / 4 = 0.525.

        Range: 0 to 1. Good value: > 0.7.
        """
        if results is None:
            raise ValueError("results must not be None")
        if not results:
            return 0.0

        average_precisions: list[float] = []
        for result in results:
            relevant_seen = 0
            precision_sum = 0.0

            for index, document in enumerate(result.retrieved_docs, start=1):
                if document.is_relevant:
                    relevant_seen += 1
                    # Precision@k = relevant documents seen so far / current rank.
                    precision_sum += relevant_seen / index

            # AP = mean precision at relevant ranks; 0 if no relevant docs retrieved.
            average_precisions.append(precision_sum / relevant_seen if relevant_seen else 0.0)

        # MAP = mean AP across all queries.
        return sum(average_precisions) / len(average_precisions)

    def calculate_ndcg(self, result: SearchResult, k: int | None = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG).

        Algorithm:
        NDCG answers: how good is ranking when position matters a lot?
        DCG discounts lower-ranked documents using ``log2(position + 1)``.
        Ideal DCG is the DCG of a perfect ranking sorted by relevance
        descending. NDCG normalizes actual DCG by ideal DCG.

        Formula:
        DCG = sum(relevance_score / log2(position + 1))
        NDCG = DCG_actual / DCG_ideal

        Worked example:
        Actual ranking relevances: [0, 1, 0, 1, 1].
        Ideal ranking relevances:  [1, 1, 1, 0, 0].
        DCG_actual = 0/log2(2) + 1/log2(3) + 0/log2(4)
                     + 1/log2(5) + 1/log2(6) = 1.45.
        DCG_ideal = 1/log2(2) + 1/log2(3) + 1/log2(4) = 2.13.
        NDCG = 1.45 / 2.13 = 0.68.

        Range: 0 to 1. Good value: > 0.8.
        """
        if result is None:
            raise ValueError("result must not be None")
        if k is not None and k <= 0:
            raise ValueError("k must be greater than 0 when provided")
        if not result.retrieved_docs:
            return 0.0

        documents = result.retrieved_docs[:k] if k is not None else result.retrieved_docs
        relevance_values = [
            document.relevance_score if document.relevance_score > 0 else (1.0 if document.is_relevant else 0.0)
            for document in documents
        ]

        def dcg(values: list[float]) -> float:
            total = 0.0
            for index, relevance in enumerate(values, start=1):
                # Discounted gain = relevance / log2(rank + 1).
                total += relevance / math.log2(index + 1)
            return total

        actual_dcg = dcg(relevance_values)
        ideal_dcg = dcg(sorted(relevance_values, reverse=True))

        # NDCG = actual ranking value / perfect ranking value.
        return actual_dcg / ideal_dcg if ideal_dcg else 0.0

    def evaluate(self, results: list[SearchResult]) -> SearchMetrics:
        """Run all search metrics and return a ``SearchMetrics`` object.

        Algorithm:
        1. Calculate MRR and MAP across all queries.
        2. Calculate precision, recall, and F1 per query.
        3. Average precision, recall, F1, and NDCG across queries.

        Formula:
        mean_metric = sum(metric_i for each query i) / number_of_queries

        Worked example:
        If two queries have precision 1.0 and 0.5, aggregate precision is
        (1.0 + 0.5) / 2 = 0.75.
        """
        if results is None:
            raise ValueError("results must not be None")
        if not results:
            return SearchMetrics(mrr=0.0, map_score=0.0, precision=0.0, recall=0.0, f1=0.0, ndcg=0.0)

        precision_recall_f1 = [self.calculate_precision_recall_f1(result) for result in results]
        ndcg_values = [self.calculate_ndcg(result) for result in results]
        query_count = len(results)

        # Average each per-query metric across all queries.
        precision = sum(values[0] for values in precision_recall_f1) / query_count
        recall = sum(values[1] for values in precision_recall_f1) / query_count
        f1 = sum(values[2] for values in precision_recall_f1) / query_count
        ndcg = sum(ndcg_values) / query_count

        return SearchMetrics(
            mrr=self.calculate_mrr(results),
            map_score=self.calculate_map(results),
            precision=precision,
            recall=recall,
            f1=f1,
            ndcg=ndcg,
        )