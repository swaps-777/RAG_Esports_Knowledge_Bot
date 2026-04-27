"""Answer quality metrics for generated RAG responses."""

from __future__ import annotations

from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_evaluator.models import AnswerEvalInput, AnswerMetrics
from rag_evaluator.utils import cosine_sim

try:  # Prefer the maintained fuzzy matching package.
    from thefuzz import fuzz
except ImportError:  # pragma: no cover - compatibility fallback.
    from fuzzywuzzy import fuzz  # type: ignore[no-redef]


class AnswerEvaluator:
    """Evaluate generated answers against a reference answer."""

    def calculate_rouge(self, generated: str, reference: str) -> dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

        Algorithm:
        ROUGE measures text overlap between generated and reference answers.
        ROUGE-1 uses unigrams, ROUGE-2 uses bigrams, and ROUGE-L uses the
        longest common subsequence, which preserves word order without requiring
        words to be consecutive.

        Formulas:
        recall = matching_units / reference_unit_count
        precision = matching_units / generated_unit_count
        F1 = 2 * (precision * recall) / (precision + recall)

        Worked example:
        Reference: "Apple reported strong quarterly earnings".
        Generated: "Apple announced strong quarterly results".
        ROUGE-1 matches Apple, strong, quarterly = 3/5 recall and 3/5
        precision, so F1 = 0.6. ROUGE-2 matches "strong quarterly" = 1/4,
        so F1 = 0.25.

        Returns keys: ``rouge1_f``, ``rouge2_f``, ``rougeL_f``.
        """
        if generated is None or reference is None:
            raise ValueError("generated and reference must not be None")
        if not generated.strip() or not reference.strip():
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, generated)

        # Each rouge-score object includes precision, recall, and F1; expose F1.
        return {
            "rouge1_f": scores["rouge1"].fmeasure,
            "rouge2_f": scores["rouge2"].fmeasure,
            "rougeL_f": scores["rougeL"].fmeasure,
        }

    def calculate_fuzzy_match(self, generated: str, reference: str) -> dict[str, float]:
        """Calculate full and partial fuzzy string matching scores.

        Algorithm:
        Fuzzy matching uses Levenshtein-style edit distance. It counts the
        minimum insertions, deletions, and replacements needed to transform one
        string into the other. ``fuzz.ratio`` compares whole strings, while
        ``fuzz.partial_ratio`` finds the best matching substring.

        Formula:
        similarity = 1 - (edit_distance / max_length), scaled to 0-100.

        Worked example:
        Reference: "Revenue was $85.8 billion".
        Generated: "In Q3 2024, Apple's revenue was $85.8 billion, up 5% YoY."
        Full ratio is lower because of extra context, but partial ratio is high
        because the key substring is present.

        Returns keys: ``fuzzy_ratio`` and ``fuzzy_partial_ratio``.
        """
        if generated is None or reference is None:
            raise ValueError("generated and reference must not be None")
        if not generated.strip() or not reference.strip():
            return {"fuzzy_ratio": 0.0, "fuzzy_partial_ratio": 0.0}

        return {
            # Full-string similarity on a 0-100 scale.
            "fuzzy_ratio": float(fuzz.ratio(generated, reference)),
            # Best-substring similarity on a 0-100 scale.
            "fuzzy_partial_ratio": float(fuzz.partial_ratio(generated, reference)),
        }

    def calculate_semantic_similarity(
        self,
        text_a: str,
        text_b: str,
        embeddings: list[list[float]] | None = None,
    ) -> float:
        """Calculate semantic similarity with embeddings or TF-IDF fallback.

        Algorithm:
        Semantic similarity asks whether two texts mean the same thing even
        when their words differ. If embeddings are provided, this method expects
        two vectors and calculates cosine similarity directly. If embeddings are
        not provided, it uses scikit-learn ``TfidfVectorizer`` as an API-free
        fallback.

        Formula:
        cosine(A, B) = dot(A, B) / (||A|| * ||B||)

        Worked example:
        "The stock price has increased" and "The shares went up" may have low
        word overlap, but embedding cosine similarity can be about 0.92 because
        the meanings are close.

        Range: -1 to 1 for arbitrary vectors, typically 0 to 1 for text vectors.
        Good value: > 0.85.
        """
        if text_a is None or text_b is None:
            raise ValueError("text_a and text_b must not be None")
        if embeddings is not None:
            if len(embeddings) != 2:
                raise ValueError("embeddings must contain exactly two vectors")
            return cosine_sim(embeddings[0], embeddings[1])
        if not text_a.strip() or not text_b.strip():
            return 0.0

        vectorizer = TfidfVectorizer().fit([text_a, text_b])
        vectors = vectorizer.transform([text_a, text_b]).toarray()

        # Cosine = dot(TF-IDF text A, TF-IDF text B) / product of vector norms.
        return cosine_sim(vectors[0].tolist(), vectors[1].tolist())

    def evaluate(self, eval_input: AnswerEvalInput) -> AnswerMetrics:
        """Run all answer metrics and return an ``AnswerMetrics`` object.

        Algorithm:
        1. Compare generated and reference answers with ROUGE overlap metrics.
        2. Compare strings with typo-tolerant fuzzy matching.
        3. Compare meaning using embedding cosine similarity, falling back to
           TF-IDF vectors when external embeddings are not supplied.

        Formula:
        The returned object is the union of ROUGE F1 scores, fuzzy 0-100 scores,
        and semantic cosine similarity.

        Worked example:
        If generated and reference answers are identical, fuzzy ratio should be
        100 and semantic similarity should be approximately 1.0.
        """
        if eval_input is None:
            raise ValueError("eval_input must not be None")

        rouge = self.calculate_rouge(eval_input.generated_answer, eval_input.reference_answer)
        fuzzy_scores = self.calculate_fuzzy_match(eval_input.generated_answer, eval_input.reference_answer)
        semantic_similarity = self.calculate_semantic_similarity(
            eval_input.generated_answer,
            eval_input.reference_answer,
        )

        return AnswerMetrics(
            rouge1_f=rouge["rouge1_f"],
            rouge2_f=rouge["rouge2_f"],
            rougeL_f=rouge["rougeL_f"],
            fuzzy_ratio=fuzzy_scores["fuzzy_ratio"],
            fuzzy_partial_ratio=fuzzy_scores["fuzzy_partial_ratio"],
            semantic_similarity=semantic_similarity,
        )