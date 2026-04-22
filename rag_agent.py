"""
game_esports_knowledge_agent.py style flow inside rag_agent.py

This module implements a health and fitness RAG agent using LangGraph in a
teaching-friendly pattern that mirrors a common classroom example:

    START
      |
    understand_question
      |
    search_index
      |
      +---> game_manual_specialist ---+
      |                               |
      +---> game_strategy_specialist -+---> pick_response_mode
      |                               |            |
      +---> esports_specialist -------+      (conditional)
                                               /         \
                                          quick        detailed
                                            |             |
                                     quick_answer   detailed_answer
                                            |             |
                                           END           END

WHAT THIS TEACHES:
1. State management with Pydantic
2. A dedicated `search_index` retrieval node
3. Parallel specialist nodes
4. Fan-in into one decision node
5. Conditional routing to different response styles

PROJECT DOMAIN:
- health tips
- gym guidance
- fitness guides
"""

from __future__ import annotations

import json
import operator
import os
from typing import Annotated

from pydantic import BaseModel, ConfigDict

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ingestion import CHROMA_DB_DIR, EMBEDDING_MODEL

import dotenv
dotenv.load_dotenv()


# ==========================================================================
# CONFIGURATION - students can experiment here
# ==========================================================================

TOP_K = 4
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0


class GamingRAGState(BaseModel):
    """
    Shared state that flows through the LangGraph application.

    Students can read this class top-to-bottom to understand what data each
    node produces and consumes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_question: str = ""
    question_analysis: str = ""

    retrieved_documents: list[Document] = []
    retrieved_context: str = ""
    retrieved_sources: str = ""

    manual_view: str = ""
    strategy_view: str = ""
    esports_view: str = ""

    needs_detailed_answer: bool = False
    answer_reason: str = ""
    final_answer: str = ""

    messages: Annotated[list[str], operator.add] = []


llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, api_key=os.getenv('OPENAI_API_KEY'))


def build_embedding_model() -> HuggingFaceEmbeddings:
    """Create the local embedding model used to search Chroma."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_vector_store() -> Chroma:
    """Open the local Chroma vector database from disk."""
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(
            f"Vector database '{CHROMA_DB_DIR}/' was not found. Run ingestion first."
        )

    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=build_embedding_model(),
    )


def format_context(documents: list[Document]) -> str:
    """Combine retrieved chunks into one prompt-ready context string."""
    if not documents:
        return "No relevant context was retrieved from the index."

    return "\n\n---\n\n".join(document.page_content for document in documents)


def format_sources(documents: list[Document]) -> str:
    """Format citation metadata into a readable source list."""
    if not documents:
        return "No sources retrieved."

    formatted_sources = []
    for index, document in enumerate(documents, start=1):
        source_file = document.metadata.get("source", "Unknown source")
        page_number = document.metadata.get("page", "?")
        page_label = page_number + 1 if isinstance(page_number, int) else page_number
        formatted_sources.append(f"[{index}] {source_file} (Page {page_label})")

    return "\n".join(formatted_sources)


def understand_question(state: GamingRAGState) -> dict:
    """
    First node: interpret the user's question before retrieval.

    This gives students a clear example of a node that adds reasoning context
    before the search step.
    """
    response = llm.invoke(
        f"You are a helpful gaming strategy and esports knowledge assistant.\n"
        f"The user asked: '{state.user_question}'.\n\n"
        f"In 2-3 short sentences, explain what the user seems to want.\n"
        f"Mention whether the question is mainly about game mechanics, "
        f"strategy, player roles, esports competition, or a mix."
    )

    return {
        "question_analysis": response.content,
        "messages": [f"[understand_question] {response.content}"],
    }


def search_index(state: GamingRAGState) -> dict:
    """
    Retrieval node: search the Chroma index for relevant chunks.

    This is the key node you wanted to show your students explicitly.
    """
    vector_store = load_vector_store()
    retrieved_documents = vector_store.similarity_search(state.user_question, k=TOP_K)

    retrieved_context = format_context(retrieved_documents)
    retrieved_sources = format_sources(retrieved_documents)

    print(f"[search_index] Found {len(retrieved_documents)} chunk(s)")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    debug_path = os.path.join(BASE_DIR, "debug_retrieval.txt")

    print(f"[debug] Writing retrieval debug file to: {debug_path}", flush=True)

    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("===== USER QUESTION =====\n")
        f.write(f"{state.user_question}\n\n")

        f.write("===== RETRIEVED SOURCES =====\n")
        f.write(f"{retrieved_sources}\n\n")

        f.write("===== RETRIEVED CONTEXT =====\n")
        f.write(retrieved_context + "\n\n")

        f.write("===== INDIVIDUAL CHUNKS =====\n")
        for i, doc in enumerate(retrieved_documents, start=1):
            f.write(f"\n--- Chunk {i} ---\n")
            f.write(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
            f.write(f"Page: {doc.metadata.get('page', 'N/A')}\n")
            f.write(doc.page_content)
            f.write("\n\n")
    
    print(f"[debug] File write complete: {debug_path}", flush=True)

    return {
        "retrieved_documents": retrieved_documents,
        "retrieved_context": retrieved_context,
        "retrieved_sources": retrieved_sources,
        "messages": [f"[search_index] Retrieved {len(retrieved_documents)} chunk(s)"],
    }


def game_manual_specialist(state: GamingRAGState) -> dict:
    """Parallel node: extract gameplay mechanics, systems, and role-related information from the retrieved context."""

    response = llm.invoke(
        f"You are a game manual specialist.\n"
        f"Your job is to answer from the perspective of official game manuals.\n\n"
        f"User question: '{state.user_question}'\n\n"
        f"Retrieved context:\n{state.retrieved_context}\n\n"
        f"Using only the retrieved context:\n"
        f"- extract relevant information from both paragraphs and tables/lists\n"
        f"- if the question asks for names, classes, roles, units, mechanics, systems, or rules, list them explicitly when present\n"
        f"- if the question asks for explanation, summarize clearly from the retrieved context\n"
        f"- do not replace exact entries with a vague summary when exact entries are available\n"
        f"- do not use outside knowledge\n"
        f"- if the answer is not present in the retrieved context, say that clearly in one short sentence"
    )

    return {
        "manual_view": response.content,
        "messages": ["[game_manual_specialist] Done"],
    }


def game_strategy_specialist(state: GamingRAGState) -> dict:
    """Parallel node: extract strategy and tactical guidance from the retrieved context."""
    response = llm.invoke(
        f"You are a game strategy specialist.\n"
        f"Your job is to answer from the perspective of strategy guides and tactical gameplay analysis.\n\n"
        f"User question: '{state.user_question}'\n\n"
        f"Retrieved context:\n{state.retrieved_context}\n\n"
        f"Using only the retrieved context:\n"
        f"- extract relevant information from both paragraphs and tables/lists\n"
        f"- if the context contains named strategies, roles, categories, or structured items, list them explicitly\n"
        f"- if the question asks for an explanation, summarize the strategy insight clearly\n"
        f"- do not replace exact extracted items with a generic summary\n"
        f"- do not use outside knowledge\n"
        f"- if the answer is not present, say that clearly in one short sentence"
    )

    return {
        "strategy_view": response.content,
        "messages": ["[game_strategy_specialist] Done"],
    }


def esports_specialist(state: GamingRAGState) -> dict:
    """Parallel node: extract esports-specific insights and competitive strategies."""
    response = llm.invoke(
        f"You are an esports specialist.\n"
        f"Your job is to answer from the perspective of esports reports, tournament rules, and competitive analysis.\n\n"
        f"User question: '{state.user_question}'\n\n"
        f"Retrieved context:\n{state.retrieved_context}\n\n"
        f"Using only the retrieved context:\n"
        f"- extract relevant information from both paragraphs and tables/lists\n"
        f"- if the user asks for names, associations, stakeholders, roles, rules, formats, or other items, list them explicitly when present\n"
        f"- if the context provides explanatory paragraph text, summarize it clearly\n"
        f"- do not replace exact names or entries with a vague summary when exact entries are available\n"
        f"- do not use outside knowledge\n"
        f"- if the answer is not present in the retrieved context, say that clearly in one short sentence"
    )

    return {
        "esports_view": response.content,
        "messages": ["[esports_specialist] Done"],
    }


def pick_response_mode(state: GamingRAGState) -> dict:
    """
    Fan-in decision node.

    It decides whether the final answer should be:
    - quick: concise explanation
    - detailed: more structured coaching-style response
    """
    response = llm.invoke(
        f"You are a response planner for a gaming strategy and esports knowledge bot.\n\n"
        f"User question:\n{state.user_question}\n\n"
        f"Question understanding:\n{state.question_analysis}\n\n"
        f"GAME MANUAL VIEW:\n{state.manual_view}\n\n"
        f"GAME STRATEGY VIEW:\n{state.strategy_view}\n\n"
        f"ESPORTS VIEW:\n{state.esports_view}\n\n"
        f"Decide whether the user needs a QUICK or DETAILED answer.\n"
        f"Choose DETAILED when the user asks for:\n"
        f"- explanation of mechanics\n"
        f"- role breakdowns\n"
        f"- strategic comparisons\n"
        f"- how/why questions\n"
        f"- competitive or tournament insights\n"
        f"- multi-step reasoning\n\n"
        f"Choose QUICK for simple factual or definition-style questions.\n\n"
        f"Reply strictly as JSON and nothing else:\n"
        f'{{"needs_detailed_answer": true, "reason": "one sentence"}}'
    )

    try:
        result = json.loads(response.content)
        needs_detailed_answer = bool(result["needs_detailed_answer"])
        answer_reason = str(result["reason"])
    except (json.JSONDecodeError, KeyError, TypeError):
        needs_detailed_answer = False
        answer_reason = "Could not parse planner output, defaulting to a quick answer."

    return {
        "needs_detailed_answer": needs_detailed_answer,
        "answer_reason": answer_reason,
        "messages": [f"[pick_response_mode] detailed={needs_detailed_answer}"],
    }


def quick_answer(state: GamingRAGState) -> dict:
    """Create a short answer for straightforward questions."""
    response = llm.invoke(
        f"You are a helpful gaming strategy and esports knowledge assistant.\n"
        f"Use only the provided specialist views and retrieved sources.\n\n"
        f"User question:\n{state.user_question}\n\n"
        f"GAME MANUAL VIEW:\n{state.manual_view}\n\n"
        f"GAME STRATEGY VIEW:\n{state.strategy_view}\n\n"
        f"ESPORTS VIEW:\n{state.esports_view}\n\n"
        f"SOURCES:\n{state.retrieved_sources}\n\n"
        f"Instructions:\n"
        f"- Use only the provided context\n"
        f"- Extract information from both paragraph text and table/list text\n"
        f"- If the question asks for names, items, roles, rules, or categories, list them explicitly when available\n"
        f"- If the question asks for an explanation, answer in a short clear summary\n"
        f"- Do not replace exact extracted entries with a generic summary\n"
        f"- If the answer is not present, say so clearly\n\n"
        f"Write a concise, beginner-friendly answer.\n"
        f"End with:\n"
        f"Sources:\n"
    )

    return {
        "final_answer": response.content,
        "messages": ["[quick_answer] Generated quick answer"],
    }


def detailed_answer(state: GamingRAGState) -> dict:
    """Create a more structured coaching-style answer for deeper questions."""
    response = llm.invoke(
        f"You are a gaming strategy and esports knowledge assistant.\n"
        f"Use only the retrieved material below.\n\n"
        f"User question:\n{state.user_question}\n\n"
        f"Question understanding:\n{state.question_analysis}\n\n"
        f"GAME MANUAL VIEW:\n{state.manual_view}\n\n"
        f"GAME STRATEGY VIEW:\n{state.strategy_view}\n\n"
        f"ESPORTS VIEW:\n{state.esports_view}\n\n"
        f"SOURCES:\n{state.retrieved_sources}\n\n"
        f"Write a clear, structured, student-friendly answer with these sections:\n"
        f"1. Main Answer\n"
        f"2. Key Extracted Details\n"
        f"3. Limits or Missing Information\n"
        f"4. Sources\n\n"
        f"Rules:\n"
        f"- Use only the provided context\n"
        f"- Extract information from both paragraphs and tables/lists\n"
        f"- If the user asks for names, rules, roles, associations, or items, list them explicitly when available\n"
        f"- If the question asks for explanation, summarize clearly\n"
        f"- Do not replace exact extracted entries with a vague summary\n"
        f"- Do not invent facts\n"
        f"- If the answer is not present, say that clearly"
    )

    return {
        "final_answer": response.content,
        "messages": ["[detailed_answer] Generated detailed answer"],
    }


def route_after_decision(state: GamingRAGState) -> str:
    """Conditional router after the planner node."""
    if state.needs_detailed_answer:
        return "detailed"
    return "quick"


def build_gaming_esports_knowledge_agent():
    """
    Build and compile the LangGraph application.

    Graph structure:
        START -> understand_question -> search_index
              -> game_manual_specialist
              -> game_strategy_specialist
              -> esports_specialist
              -> pick_response_mode
              -> quick_answer OR detailed_answer
              -> END
    """
    graph = StateGraph(GamingRAGState)

    graph.add_node("understand_question", understand_question)
    graph.add_node("search_index", search_index)
    graph.add_node("game_manual_specialist", game_manual_specialist)
    graph.add_node("game_strategy_specialist", game_strategy_specialist)
    graph.add_node("esports_specialist", esports_specialist)
    graph.add_node("pick_response_mode", pick_response_mode)
    graph.add_node("quick_answer", quick_answer)
    graph.add_node("detailed_answer", detailed_answer)

    graph.add_edge(START, "understand_question")
    graph.add_edge("understand_question", "search_index")

    graph.add_edge("search_index", "game_manual_specialist")
    graph.add_edge("search_index", "game_strategy_specialist")
    graph.add_edge("search_index", "esports_specialist")

    graph.add_edge("game_manual_specialist", "pick_response_mode")
    graph.add_edge("game_strategy_specialist", "pick_response_mode")
    graph.add_edge("esports_specialist", "pick_response_mode")

    graph.add_conditional_edges(
        "pick_response_mode",
        route_after_decision,
        {
            "quick": "quick_answer",
            "detailed": "detailed_answer",
        },
    )

    graph.add_edge("quick_answer", END)
    graph.add_edge("detailed_answer", END)

    return graph.compile()


def query_rag(question: str) -> str:
    """Run one user question through the Gaming and Esports Knowledge LangGraph agent."""
    app = build_gaming_esports_knowledge_agent()
    result = app.invoke(
        {
            "user_question": question,
            "messages": [],
        }
    )
    return result["final_answer"]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    answer = query_rag("What do these documents say about building a beginner workout routine?")
    print()
    print(answer)
