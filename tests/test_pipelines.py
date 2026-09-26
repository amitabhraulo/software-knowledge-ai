"""RAG behavior tests: no model download, network, or Groq key required."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from main import parse_args
from src.rag.pipelines import (
    check_citations, compare_question, evidence_context, format_result, run_pipeline,
)
from src.search.scored_retriever import retrieve_scored


def fake_store(pairs=None, metric="l2"):
    store = Mock()
    store._collection.configuration = {"hnsw": {"space": metric}}
    store.similarity_search_with_score.return_value = pairs if pairs is not None else [
        (Document(page_content="A gateway routes requests.", metadata={"source": "a.pdf", "page": 0}), 0.2),
        (Document(page_content="A saga compensates transactions.", metadata={"source": "b.pdf", "page": 5}), 1.2),
    ]
    return store


class RetrievalTests(unittest.TestCase):
    def test_cutoff_includes_boundary_and_filters_use_and(self):
        store = fake_store()
        result = retrieve_scored("gateway", store=store, source="a.pdf", category="architecture", max_distance=0.2)
        self.assertEqual([h.accepted for h in result.hits], [True, False])
        self.assertEqual(result.metric, "l2")
        store.similarity_search_with_score.assert_called_once_with(
            "gateway", k=5, filter={"$and": [{"source": "a.pdf"}, {"category": "architecture"}]})

    def test_unknown_metric_and_nonfinite_cutoffs_fail(self):
        with self.assertRaises(ValueError):
            retrieve_scored("gateway", store=fake_store(metric="unknown"))
        for cutoff in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                retrieve_scored("gateway", store=fake_store(), max_distance=cutoff)


class PipelineTests(unittest.TestCase):
    def test_empty_or_rejected_evidence_skips_llm(self):
        for store, cutoff in ((fake_store([]), None), (fake_store(), 0.1)):
            with patch("src.rag.pipelines.get_llm") as factory:
                result = run_pipeline("gateway", mode="improved", store=store, max_distance=cutoff)
                factory.assert_not_called()
                self.assertFalse(result.llm_called)
                self.assertEqual(result.generation_seconds, 0)

    def test_compare_preserves_baseline_and_limits_improved_context(self):
        prompts = []
        def answer(prompt):
            prompts.append(prompt.to_messages())
            return AIMessage(content="A gateway routes requests. [S1]")
        results = compare_question("gateway", store=fake_store(),
                                   max_distance=0.3, llm=RunnableLambda(answer))
        self.assertEqual(len(results[0].retrieval.documents), 2)
        self.assertEqual(len(results[1].retrieval.documents), 1)
        self.assertEqual([m.type for m in prompts[0]], ["human"])
        self.assertIn("senior software architecture assistant", prompts[0][0].content)
        self.assertIn("saga", prompts[0][0].content)
        self.assertEqual([m.type for m in prompts[1]], ["system", "human"])
        self.assertNotIn("saga", prompts[1][1].content)
        self.assertIn("[S1]", prompts[1][1].content)
        report = format_result(results[1], show_context=True)
        self.assertIn("squared L2 distance", report)
        self.assertIn("[rejected]", report)
        self.assertIn("not verified", results[1].citation_status)

    def test_citation_failures_are_visible(self):
        self.assertIn("Invalid", check_citations("Claim [S3]", 2))
        self.assertIn("Missing", check_citations("Claim without citation", 2))
        self.assertIn("no citations expected", check_citations("I could not find this in the indexed documents.", 2))

    def test_page_labels_and_display_numbers(self):
        docs = [Document(page_content="Text", metadata={"source": "a.pdf", "page": 0}),
                Document(page_content="Other", metadata={"source": "b.pdf", "page": 9, "page_label": "iv"})]
        context = evidence_context(docs)
        self.assertIn("Page: 1", context)
        self.assertIn("Page: iv", context)

    def test_baseline_rejects_improved_only_options(self):
        with self.assertRaises(ValueError):
            run_pipeline("gateway", mode="baseline", category="general", store=fake_store())

    def test_cli_defaults_and_compare_options(self):
        self.assertEqual(parse_args([]).mode, "baseline")
        args = parse_args(["--mode", "compare", "--max-distance", "0.5", "--question", "gateway"])
        self.assertEqual(args.max_distance, 0.5)
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parse_args(["--source", "a.pdf"])


if __name__ == "__main__":
    unittest.main()
