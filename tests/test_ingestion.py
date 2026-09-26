"""Run with python -m unittest discover -s tests -v (MiniLM tokenizer cached)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from transformers import AutoTokenizer

from src.config import EMBEDDING_MODEL_NAME
from src.ingest import chunking, enterprise_ingest as ingest


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, local_files_only=True)

    def test_long_text_preserves_metadata_and_fits_model(self):
        text = ("Distributed transactions use compensation.\n" * 200
                + "\n\n" + "someLongIdentifier.method(arg); " * 200)
        with patch.object(chunking, "get_tokenizer", return_value=self.tokenizer):
            chunks = chunking.split_documents([Document(page_content=text, metadata={"page": 7, "source": "test.pdf"})])
        self.assertGreater(len(chunks), 1)
        self.assertIn("someLongIdentifier", chunks[-1].page_content)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["page"], 7)
            self.assertEqual(chunk.metadata["source"], "test.pdf")
            self.assertLessEqual(len(self.tokenizer.encode(chunk.page_content)), 256)
            self.assertLessEqual(chunk.metadata["token_count"], 240)

    def test_invalid_budget_is_rejected(self):
        with patch.object(chunking, "get_tokenizer", return_value=self.tokenizer), patch.object(chunking, "CHUNK_SIZE", 256):
            with self.assertRaises(ValueError):
                chunking.split_documents([Document(page_content="hello")])


class MemoryStore:
    def __init__(self):
        self._collection = self
        self.documents = []
        self.fail = False

    def delete(self, where):
        self.documents = [d for d in self.documents if d.metadata["file_path"] != where["file_path"]]

    def add_documents(self, documents):
        self.documents.extend(documents)
        if self.fail:
            raise RuntimeError("Simulated failure after a partial write")


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / "documents"
        self.docs.mkdir()
        self.store = MemoryStore()
        self.patches = [
            patch.object(ingest, "DOCUMENT_DIR", self.docs),
            patch.object(ingest, "VECTOR_DB_DIR", self.root / "db"),
            patch.object(ingest, "MANIFEST_FILE", self.root / "db" / "manifest.json"),
            patch.object(ingest, "get_vector_store", return_value=self.store),
            patch.object(ingest, "load_pdf_documents", side_effect=lambda path, digest: [Document(
                page_content=path.read_text(), metadata={"file_path": str(path), "file_hash": digest}
            )]),
            patch.object(ingest, "split_documents", side_effect=lambda docs: docs),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_unchanged_skips_and_identical_files_update_independently(self):
        first, second = self.docs / "first.pdf", self.docs / "second.pdf"
        first.write_text("same")
        second.write_text("same")
        ingest.ingest_documents_incremental()
        ingest.ingest_documents_incremental()
        self.assertEqual(len(self.store.documents), 2)
        first.write_text("updated")
        ingest.ingest_documents_incremental()
        self.assertCountEqual([d.page_content for d in self.store.documents], ["updated", "same"])

    def test_retry_after_partial_write_does_not_duplicate_chunks(self):
        path = self.docs / "first.pdf"
        path.write_text("original")
        ingest.ingest_documents_incremental()
        path.write_text("changed")
        self.store.fail = True
        with self.assertRaises(RuntimeError):
            ingest.ingest_documents_incremental()
        self.assertNotIn("first.pdf", ingest.load_manifest())
        self.store.fail = False
        ingest.ingest_documents_incremental()
        self.assertEqual([d.page_content for d in self.store.documents], ["changed"])

    def test_new_configuration_manifest_reindexes_unchanged_source(self):
        (self.docs / "first.pdf").write_text("same")
        ingest.ingest_documents_incremental()
        new_store = MemoryStore()
        with patch.object(ingest, "MANIFEST_FILE", self.root / "db" / "new-config.json"), patch.object(ingest, "get_vector_store", return_value=new_store):
            ingest.ingest_documents_incremental()
            self.assertEqual(len(new_store.documents), 1)
        self.assertEqual(len(self.store.documents), 1)


if __name__ == "__main__":
    unittest.main()
