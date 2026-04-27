"""
main.py - Interactive question-answering app for the RAG project.

Usage:
1. Put PDF files inside `data/`
2. Run `python ingestion.py`
3. Run `python main.py`
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from ingestion import CHROMA_DB_DIR
from rag_agent import query_rag


load_dotenv()


def print_banner() -> None:
    """Display a simple welcome banner."""
    print("=" * 60)
    print("RAG AI Agent - LangGraph + ChromaDB")
    print("=" * 60)


def vector_database_exists() -> bool:
    """Check whether ingestion has already created the Chroma database."""
    return os.path.exists(CHROMA_DB_DIR)


def print_setup_instructions() -> None:
    """Tell the user how to prepare the project if ingestion has not been run."""
    print("No vector database found.")
    print()
    print("Run these steps first:")
    print("1. Put PDF files inside the 'data/' folder")
    print("2. Run: python ingestion.py")
    print("3. Then run: python main.py")


def run_chat_loop() -> None:
    """Start the interactive question-answering loop."""
    print()
    print("Ask a question about your documents.")
    print("Type 'quit' to exit.")
    print()

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return

        answer = query_rag(question)
        print()
        print(f"Assistant: {answer}")
        print()
        print("-" * 60)


def main() -> None:
    """Run the CLI application."""
    print_banner()
    print()

    if not vector_database_exists():
        print_setup_instructions()
        return

    print(f"Using vector database at '{CHROMA_DB_DIR}/'.")
    run_chat_loop()


if __name__ == "__main__":
    main()
