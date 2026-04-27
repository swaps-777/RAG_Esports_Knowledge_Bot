"""LLM-as-a-judge evaluators for groundedness, precision, and relevancy."""

from __future__ import annotations

import json
from typing import Any

from rag_evaluator.models import LLMJudgeMetrics


GROUNDING_PROMPT_TEMPLATE = """You are an evaluation judge. Given the following answer and a source document,
determine if the source directly supports the answer.

Answer: {answer}
Source: {source}

Rate the source:
- "useful_1" if the source directly supports claims in the answer
- "useful_0" if the source is not relevant or too generic

Respond with ONLY "useful_1" or "useful_0" followed by a brief reason.
"""

PRECISION_PROMPT_TEMPLATE = """You are a fact-checking judge. Decompose the answer into individual claims,
then classify each claim against the provided sources.

Answer: {answer}
Sources: {sources_joined}

For each claim, classify as:
- TP: claim is supported by the sources
- FP: claim is NOT supported by sources (hallucination)
- FN: important info in sources but missing from answer

Respond as JSON:
{{
  "claims": [
    {{"text": "claim text", "classification": "TP|FP|FN", "reason": "why"}}
  ],
  "tp_count": N,
  "fp_count": N,
  "fn_count": N
}}
"""

RELEVANCY_PROMPT_TEMPLATE = """You are a relevancy judge. Determine if the answer directly and helpfully
addresses the user's query, or if it is vague/evasive/non-committal.

Query: {query}
Answer: {answer}

Evaluate:
1. Does the answer directly address the query? (yes/no)
2. Is the answer non-committal or evasive? (yes/no)
3. Relevancy score (0-1, where 1 = perfectly relevant and direct)

Respond as JSON:
{{
  "directly_addresses_query": true/false,
  "is_noncommittal": true/false,
  "relevancy_score": 0.0-1.0,
  "reason": "explanation"
}}
"""


class LLMJudge:
    """Evaluate RAG answers with an OpenAI-compatible LLM judge."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None) -> None:
        """Create an LLM judge client.

        Algorithm:
        Lazily import the optional OpenAI dependency and create a client that can
        target either OpenAI or an OpenAI-compatible internal endpoint.

        Formula-style view:
        client = OpenAI(api_key=api_key, base_url=base_url)

        Worked example:
        ``LLMJudge(model="gpt-4o", base_url="https://internal-bank-llm")``
        sends chat-completions requests to the internal endpoint.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional install.
            raise ImportError(
                "LLMJudge requires the optional OpenAI dependency. "
                "Install it with `pip install rag-evaluator[llm]` or `pip install openai`."
            ) from exc

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return str(response.choices[0].message.content or "")

    @staticmethod
    def _parse_json_response(raw_text: str) -> dict[str, Any]:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"LLM response did not contain JSON: {raw_text}")
        return dict(json.loads(raw_text[start : end + 1]))

    def check_grounding(self, answer: str, sources: list[str]) -> dict[str, Any]:
        """Check whether each source directly supports the answer.

        Algorithm:
        For each source, ask the LLM judge whether that source is useful for the
        final answer. The judge must return ``useful_1`` for direct support or
        ``useful_0`` for irrelevant or generic content. Grounding percentage is
        useful sources divided by total sources, multiplied by 100.

        Formula:
        grounding_score = (useful_source_count / total_source_count) * 100

        Worked example:
        Answer: "Apple's Q3 revenue was $85.8B, up 5% YoY."
        Source 1 supports the fact, Source 2 is about iPhone, Source 3 is a
        generic tech-sector note. Useful count = 1, total = 3, so grounding
        score = (1/3) * 100 = 33.33.
        """
        if answer is None or sources is None:
            raise ValueError("answer and sources must not be None")
        if not sources:
            return {"grounding_score": 0.0, "useful_count": 0, "total_sources": 0, "source_results": []}

        source_results: list[dict[str, Any]] = []
        useful_count = 0
        for source in sources:
            prompt = GROUNDING_PROMPT_TEMPLATE.format(answer=answer, source=source)
            raw_response = self._complete(prompt).strip()
            is_useful = raw_response.lower().startswith("useful_1")
            if is_useful:
                useful_count += 1
            source_results.append({"source": source, "is_useful": is_useful, "raw_response": raw_response})

        # Grounding percentage = useful sources / all sources * 100.
        grounding_score = (useful_count / len(sources)) * 100.0
        return {
            "grounding_score": grounding_score,
            "useful_count": useful_count,
            "total_sources": len(sources),
            "source_results": source_results,
        }

    def check_precision(self, answer: str, sources: list[str]) -> dict[str, Any]:
        """Check answer claims for TP, FP, and FN hallucination categories.

        Algorithm:
        Ask the LLM to decompose the answer into claims and classify each claim
        against the sources. TP means supported, FP means unsupported and
        therefore hallucinated, and FN means important source information was
        omitted from the answer.

        Formula:
        precision_score = TP / (TP + FP) * 100
        hallucination_count = FP

        Worked example:
        Source: "Apple Q3 revenue: $85.8B".
        Generated: "Apple reported Q3 revenue of $85.8B. Mac sales declined by
        3%." Revenue is TP=1; Mac sales is FP=1. Precision = 1/(1+1)*100 = 50.
        """
        if answer is None or sources is None:
            raise ValueError("answer and sources must not be None")

        sources_joined = "\n\n".join(sources)
        prompt = PRECISION_PROMPT_TEMPLATE.format(answer=answer, sources_joined=sources_joined)
        parsed = self._parse_json_response(self._complete(prompt))

        tp_count = int(parsed.get("tp_count", 0))
        fp_count = int(parsed.get("fp_count", 0))
        fn_count = int(parsed.get("fn_count", 0))

        # Precision percentage = supported claims / all asserted claims * 100.
        precision_score = (tp_count / (tp_count + fp_count)) * 100.0 if tp_count + fp_count else 0.0
        parsed.update(
            {
                "precision_score": precision_score,
                "hallucination_count": fp_count,
                "has_hallucination": fp_count > 0,
                "tp_count": tp_count,
                "fp_count": fp_count,
                "fn_count": fn_count,
            }
        )
        return parsed

    def check_relevancy(self, query: str, answer: str) -> dict[str, Any]:
        """Check whether the answer directly addresses the query.

        Algorithm:
        Ask the LLM judge whether the answer is direct and helpful, whether it
        is non-committal or evasive, and what relevancy score it deserves from
        0 to 1. This catches answers like "it depends" when the user asked for a
        specific fact.

        Formula:
        relevancy_score in [0, 1], where 1 means direct and useful.
        is_noncommittal = true when the answer is evasive or vague.

        Worked example:
        Query: "What was Apple's Q3 revenue?"
        Answer A: "Apple's Q3 2024 revenue was $85.8 billion." returns
        noncommittal=false and relevancy=1.0.
        Answer B: "Apple's revenue varies by quarter" returns
        noncommittal=true and a low relevancy score.
        """
        if query is None or answer is None:
            raise ValueError("query and answer must not be None")

        prompt = RELEVANCY_PROMPT_TEMPLATE.format(query=query, answer=answer)
        parsed = self._parse_json_response(self._complete(prompt))
        parsed["relevancy_score"] = float(parsed.get("relevancy_score", 0.0))
        parsed["is_noncommittal"] = bool(parsed.get("is_noncommittal", False))
        return parsed

    def evaluate(self, query: str, answer: str, sources: list[str]) -> LLMJudgeMetrics:
        """Run all LLM judge checks and return ``LLMJudgeMetrics``.

        Algorithm:
        1. Grounding check scores whether source chunks support the answer.
        2. Precision check identifies supported and hallucinated claims.
        3. Relevancy check detects vague or evasive answers.

        Formula:
        grounding_score = useful_sources / total_sources * 100
        precision_score = TP / (TP + FP) * 100
        hallucination_count = FP

        Worked example:
        If one of three sources is useful and one of two claims is unsupported,
        grounding is 33.33, precision is 50.0, and hallucination_count is 1.
        """
        if query is None or answer is None or sources is None:
            raise ValueError("query, answer, and sources must not be None")

        grounding = self.check_grounding(answer, sources)
        precision = self.check_precision(answer, sources)
        relevancy = self.check_relevancy(query, answer)

        return LLMJudgeMetrics(
            grounding_score=float(grounding["grounding_score"]),
            precision_score=float(precision["precision_score"]),
            hallucination_count=int(precision["hallucination_count"]),
            relevancy_score=float(relevancy["relevancy_score"]),
            is_noncommittal=bool(relevancy["is_noncommittal"]),
            details={"grounding": grounding, "precision": precision, "relevancy": relevancy},
        )