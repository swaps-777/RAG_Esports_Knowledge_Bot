import io
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import ingestion
import main
import rag_agent


TEST_WORKSPACE = Path("tests") / "_tmp"


class IngestionTests(unittest.TestCase):
    def setUp(self):
        TEST_WORKSPACE.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if TEST_WORKSPACE.exists():
            shutil.rmtree(TEST_WORKSPACE, ignore_errors=True)

    def test_ensure_data_directory_creates_missing_directory(self):
        missing_dir = TEST_WORKSPACE / "new_data"

        result = ingestion.ensure_data_directory(str(missing_dir))

        self.assertFalse(result)
        self.assertTrue(missing_dir.exists())

    def test_get_pdf_paths_returns_sorted_pdf_files_only(self):
        data_dir = TEST_WORKSPACE / "pdfs"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "b.pdf").write_text("x", encoding="utf-8")
        (data_dir / "a.PDF").write_text("x", encoding="utf-8")
        (data_dir / "notes.txt").write_text("ignore", encoding="utf-8")

        pdf_paths = ingestion.get_pdf_paths(str(data_dir))

        self.assertEqual(
            pdf_paths,
            [
                str(data_dir / "a.PDF"),
                str(data_dir / "b.pdf"),
            ],
        )

    @patch("ingestion.PyPDFLoader")
    def test_load_pdf_documents_uses_loader_for_each_pdf(self, mock_loader_class):
        mock_loader_a = MagicMock()
        mock_loader_a.load.return_value = [Document(page_content="doc a")]
        mock_loader_b = MagicMock()
        mock_loader_b.load.return_value = [Document(page_content="doc b")]
        mock_loader_class.side_effect = [mock_loader_a, mock_loader_b]

        documents = ingestion.load_pdf_documents(["one.pdf", "two.pdf"])

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].page_content, "doc a")
        self.assertEqual(documents[1].page_content, "doc b")

    def test_split_documents_preserves_content_in_chunks(self):
        document = Document(page_content="A" * 1200, metadata={"source": "sample.pdf", "page": 0})

        chunks = ingestion.split_documents([document])

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(isinstance(chunk, Document) for chunk in chunks))
        self.assertEqual(chunks[0].metadata["source"], "sample.pdf")

    @patch("ingestion.Chroma.from_documents")
    @patch("ingestion.build_embedding_model")
    def test_create_vector_store_calls_chroma_with_expected_arguments(
        self, mock_build_embedding_model, mock_from_documents
    ):
        chunks = [Document(page_content="chunk text")]
        fake_embeddings = object()
        fake_store = object()
        mock_build_embedding_model.return_value = fake_embeddings
        mock_from_documents.return_value = fake_store

        vector_store = ingestion.create_vector_store(chunks)

        self.assertIs(vector_store, fake_store)
        mock_from_documents.assert_called_once_with(
            documents=chunks,
            embedding=fake_embeddings,
            persist_directory=ingestion.CHROMA_DB_DIR,
        )

    @patch("ingestion.create_vector_store")
    @patch("ingestion.split_documents")
    @patch("ingestion.load_source_documents")
    def test_run_ingestion_returns_none_when_no_documents(
        self, mock_load_source_documents, mock_split_documents, mock_create_vector_store
    ):
        mock_load_source_documents.return_value = []

        result = ingestion.run_ingestion()

        self.assertIsNone(result)
        mock_split_documents.assert_not_called()
        mock_create_vector_store.assert_not_called()


class RagAgentTests(unittest.TestCase):
    def test_format_context_returns_fallback_for_empty_documents(self):
        context = rag_agent.format_context([])
        self.assertIn("No relevant context", context)

    def test_format_sources_formats_page_numbers_as_one_based(self):
        documents = [
            Document(page_content="one", metadata={"source": "data/file.pdf", "page": 0}),
            Document(page_content="two", metadata={"source": "data/file.pdf", "page": 4}),
        ]

        sources = rag_agent.format_sources(documents)

        self.assertIn("[1] data/file.pdf (Page 1)", sources)
        self.assertIn("[2] data/file.pdf (Page 5)", sources)

    def test_route_after_decision_routes_to_quick(self):
        state = rag_agent.HealthFitnessState(needs_detailed_answer=False)
        self.assertEqual(rag_agent.route_after_decision(state), "quick")

    def test_route_after_decision_routes_to_detailed(self):
        state = rag_agent.HealthFitnessState(needs_detailed_answer=True)
        self.assertEqual(rag_agent.route_after_decision(state), "detailed")

    def test_build_health_fitness_agent_contains_expected_nodes(self):
        app = rag_agent.build_health_fitness_agent()
        graph = app.get_graph()
        mermaid_text = graph.draw_mermaid()

        self.assertIn("understand_question", mermaid_text)
        self.assertIn("search_index", mermaid_text)
        self.assertIn("health_specialist", mermaid_text)
        self.assertIn("gym_specialist", mermaid_text)
        self.assertIn("fitness_specialist", mermaid_text)
        self.assertIn("pick_response_mode", mermaid_text)
        self.assertIn("quick_answer", mermaid_text)
        self.assertIn("detailed_answer", mermaid_text)

    @patch("rag_agent.build_health_fitness_agent")
    def test_query_rag_returns_final_answer(self, mock_build_health_fitness_agent):
        fake_app = MagicMock()
        fake_app.invoke.return_value = {"final_answer": "Test answer"}
        mock_build_health_fitness_agent.return_value = fake_app

        answer = rag_agent.query_rag("What is a beginner workout?")

        self.assertEqual(answer, "Test answer")
        fake_app.invoke.assert_called_once()

    def test_pick_response_mode_defaults_to_quick_when_json_parse_fails(self):
        state = rag_agent.HealthFitnessState(
            user_question="Help me with gym basics",
            question_analysis="The user wants beginner help.",
            health_view="Health info",
            gym_view="Gym info",
            fitness_view="Fitness info",
        )

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="not-json")

        with patch("rag_agent.llm", fake_llm):
            result = rag_agent.pick_response_mode(state)

        self.assertFalse(result["needs_detailed_answer"])
        self.assertIn("defaulting to a quick answer", result["answer_reason"])


class MainTests(unittest.TestCase):
    def test_vector_database_exists_uses_project_constant(self):
        with patch("main.os.path.exists", return_value=True) as mock_exists:
            result = main.vector_database_exists()

        self.assertTrue(result)
        mock_exists.assert_called_once_with(main.CHROMA_DB_DIR)

    @patch("main.print_setup_instructions")
    @patch("main.vector_database_exists", return_value=False)
    @patch("main.print_banner")
    def test_main_shows_setup_instructions_when_database_missing(
        self, mock_print_banner, mock_vector_database_exists, mock_print_setup_instructions
    ):
        main.main()

        mock_print_banner.assert_called_once()
        mock_vector_database_exists.assert_called_once()
        mock_print_setup_instructions.assert_called_once()

    @patch("main.run_chat_loop")
    @patch("main.vector_database_exists", return_value=True)
    @patch("main.print_banner")
    def test_main_starts_chat_loop_when_database_exists(
        self, mock_print_banner, mock_vector_database_exists, mock_run_chat_loop
    ):
        main.main()

        mock_print_banner.assert_called_once()
        mock_vector_database_exists.assert_called_once()
        mock_run_chat_loop.assert_called_once()

    @patch("main.query_rag", return_value="Mocked answer")
    @patch("builtins.input", side_effect=["What is fitness?", "quit"])
    def test_run_chat_loop_answers_then_exits(self, mock_input, mock_query_rag):
        stdout_buffer = io.StringIO()

        with redirect_stdout(stdout_buffer):
            main.run_chat_loop()

        output = stdout_buffer.getvalue()
        self.assertIn("Assistant: Mocked answer", output)
        self.assertIn("Goodbye!", output)
        mock_query_rag.assert_called_once_with("What is fitness?")


if __name__ == "__main__":
    unittest.main(verbosity=2)