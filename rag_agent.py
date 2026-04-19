"""
health_fitness_agent.py style flow inside rag_agent.py

This module implements a health and fitness RAG agent using LangGraph in a
teaching-friendly pattern that mirrors a common classroom example:

    START
      |
    understand_question
      |
    search_index
      |
      +---> health_specialist --------+
      |                               |
      +---> gym_specialist -----------+---> pick_response_mode
      |                               |            |
      +---> fitness_specialist -------+      (conditional)
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


# ==========================================================================
# CONFIGURATION - students can experiment here
# ==========================================================================

TOP_K = 4
LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0


class HealthFitnessState(BaseModel):
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

    health_view: str = ""
    gym_view: str = ""
    fitness_view: str = ""

    needs_detailed_answer: bool = False
    answer_reason: str = ""
    final_answer: str = ""

    messages: Annotated[list[str], operator.add] = []


llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)


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


def understand_question(state: HealthFitnessState) -> dict:
    """
    First node: interpret the user's question before retrieval.

    This gives students a clear example of a node that adds reasoning context
    before the search step.
    """
    response = llm.invoke(
        f"You are a helpful health and fitness assistant.\n"
        f"The user asked: '{state.user_question}'.\n\n"
        f"In 2-3 short sentences, explain what the user seems to want.\n"
        f"Mention whether the question is mainly about health, gym training, "
        f"fitness habits, or a mix."
    )

    return {
        "question_analysis": response.content,
        "messages": [f"[understand_question] {response.content}"],
    }


def search_index(state: HealthFitnessState) -> dict:
    """
    Retrieval node: search the Chroma index for relevant chunks.

    This is the key node you wanted to show your students explicitly.
    """
    vector_store = load_vector_store()
    retrieved_documents = vector_store.similarity_search(state.user_question, k=TOP_K)

    retrieved_context = format_context(retrieved_documents)
    retrieved_sources = format_sources(retrieved_documents)

    print(f"[search_index] Found {len(retrieved_documents)} chunk(s)")
    return {
        "retrieved_documents": retrieved_documents,
        "retrieved_context": retrieved_context,
        "retrieved_sources": retrieved_sources,
        "messages": [f"[search_index] Retrieved {len(retrieved_documents)} chunk(s)"],
    }


def health_specialist(state: HealthFitnessState) -> dict:
    """Parallel node: extract health and safety guidance from the retrieved context."""
    response = llm.invoke(
        f"You are a health and wellness specialist.\n"
        f"User question: '{state.user_question}'\n\n"
        f"Retrieved context:\n{state.retrieved_context}\n\n"
        f"Using only the retrieved context, summarize the most relevant health, "
        f"wellness, or safety guidance. If the context does not contain useful "
        f"health information, say that clearly in one short sentence."
    )

    return {
        "health_view": response.content,
        "messages": ["[health_specialist] Done"],
    }


def gym_specialist(state: HealthFitnessState) -> dict:
    """Parallel node: extract gym and workout-specific guidance."""
    response = llm.invoke(
        f"You are a gym training coach.\n"
        f"User question: '{state.user_question}'\n\n"
        f"Retrieved context:\n{state.retrieved_context}\n\n"
        f"Using only the retrieved context, summarize the most relevant gym, "
        f"exercise, or strength-training advice. If the context does not contain "
        f"useful gym guidance, say that clearly in one short sentence."
    )

    return {
        "gym_view": response.content,
        "messages": ["[gym_specialist] Done"],
    }


def fitness_specialist(state: HealthFitnessState) -> dict:
    """Parallel node: extract overall fitness and habit-building guidance."""
    response = llm.invoke(
        f"You are a general fitness coach.\n"
        f"User question: '{state.user_question}'\n\n"
        f"Retrieved context:\n{state.retrieved_context}\n\n"
        f"Using only the retrieved context, summarize the most relevant fitness, "
        f"routine, consistency, or goal-oriented advice. If the context does not "
        f"contain useful fitness guidance, say that clearly in one short sentence."
    )

    return {
        "fitness_view": response.content,
        "messages": ["[fitness_specialist] Done"],
    }


def pick_response_mode(state: HealthFitnessState) -> dict:
    """
    Fan-in decision node.

    It decides whether the final answer should be:
    - quick: concise explanation
    - detailed: more structured coaching-style response
    """
    response = llm.invoke(
        f"You are a response planner for a health and fitness RAG assistant.\n\n"
        f"User question:\n{state.user_question}\n\n"
        f"Question analysis:\n{state.question_analysis}\n\n"
        f"HEALTH VIEW:\n{state.health_view}\n\n"
        f"GYM VIEW:\n{state.gym_view}\n\n"
        f"FITNESS VIEW:\n{state.fitness_view}\n\n"
        f"Choose whether the user needs a QUICK answer or a DETAILED answer.\n"
        f"Use DETAILED when the user asks for a plan, routine, multi-step guidance, "
        f"comparison, or explanation. Use QUICK for straightforward questions.\n\n"
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


def quick_answer(state: HealthFitnessState) -> dict:
    """Create a short answer for straightforward questions."""
    response = llm.invoke(
        f"You are a helpful health and fitness assistant.\n"
        f"Answer the user's question using only the information below.\n\n"
        f"User question:\n{state.user_question}\n\n"
        f"HEALTH VIEW:\n{state.health_view}\n\n"
        f"GYM VIEW:\n{state.gym_view}\n\n"
        f"FITNESS VIEW:\n{state.fitness_view}\n\n"
        f"SOURCES:\n{state.retrieved_sources}\n\n"
        f"Write a concise, beginner-friendly answer in a short paragraph or a few "
        f"bullets. If the context is insufficient, say so clearly. End with:\n"
        f"Sources:\n"
    )

    return {
        "final_answer": response.content,
        "messages": ["[quick_answer] Generated quick answer"],
    }


def detailed_answer(state: HealthFitnessState) -> dict:
    """Create a more structured coaching-style answer for deeper questions."""
    response = llm.invoke(
        f"You are a supportive health and fitness coach.\n"
        f"Answer the user's question using only the information below.\n\n"
        f"User question:\n{state.user_question}\n\n"
        f"Question analysis:\n{state.question_analysis}\n\n"
        f"HEALTH VIEW:\n{state.health_view}\n\n"
        f"GYM VIEW:\n{state.gym_view}\n\n"
        f"FITNESS VIEW:\n{state.fitness_view}\n\n"
        f"SOURCES:\n{state.retrieved_sources}\n\n"
        f"Write a structured, student-friendly answer with these sections:\n"
        f"1. Main Answer\n"
        f"2. Practical Takeaways\n"
        f"3. Limits or Missing Information\n"
        f"4. Sources\n\n"
        f"If the context is insufficient, say that clearly instead of inventing details."
    )

    return {
        "final_answer": response.content,
        "messages": ["[detailed_answer] Generated detailed answer"],
    }


def route_after_decision(state: HealthFitnessState) -> str:
    """Conditional router after the planner node."""
    if state.needs_detailed_answer:
        return "detailed"
    return "quick"


def build_health_fitness_agent():
    """
    Build and compile the LangGraph application.

    Graph structure:
        START -> understand_question -> search_index
              -> health_specialist
              -> gym_specialist
              -> fitness_specialist
              -> pick_response_mode
              -> quick_answer OR detailed_answer
              -> END
    """
    graph = StateGraph(HealthFitnessState)

    graph.add_node("understand_question", understand_question)
    graph.add_node("search_index", search_index)
    graph.add_node("health_specialist", health_specialist)
    graph.add_node("gym_specialist", gym_specialist)
    graph.add_node("fitness_specialist", fitness_specialist)
    graph.add_node("pick_response_mode", pick_response_mode)
    graph.add_node("quick_answer", quick_answer)
    graph.add_node("detailed_answer", detailed_answer)

    graph.add_edge(START, "understand_question")
    graph.add_edge("understand_question", "search_index")

    graph.add_edge("search_index", "health_specialist")
    graph.add_edge("search_index", "gym_specialist")
    graph.add_edge("search_index", "fitness_specialist")

    graph.add_edge("health_specialist", "pick_response_mode")
    graph.add_edge("gym_specialist", "pick_response_mode")
    graph.add_edge("fitness_specialist", "pick_response_mode")

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
    """Run one user question through the health and fitness LangGraph agent."""
    app = build_health_fitness_agent()
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
