"""
Enterprise ingestion pipeline.

Version 1:
- Load all PDFs
- Split all documents
- Store everything

Version 2:
- Detect new/changed files using hash
- Skip already indexed files
- Process in batches
- Store metadata for future filtering

Version 3:
- If a file changed, delete old chunks before re-indexing
"""

import hashlib
import json
from datetime import datetime, timezone

from langchain_community.document_loaders import PyMuPDFLoader
from src.ingest.chunking import split_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
    DOCUMENT_DIR,
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    INDEX_CONFIG,
    MANIFEST_FILE,
    BATCH_SIZE,
)


def get_file_hash(file_path):
    """
    Creates SHA256 hash for a file.

    If file content changes, this hash changes.
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def load_manifest():
    """
    Loads index_manifest.json.

    The manifest stores file indexing status.
    """
    if not MANIFEST_FILE.exists():
        return {}

    with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_manifest(manifest):
    """
    Saves indexing status back to index_manifest.json.
    """
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    temporary_file = MANIFEST_FILE.with_suffix(".tmp")
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    temporary_file.replace(MANIFEST_FILE)


def get_pdf_category(pdf_file):
    """
    Uses the parent folder as document category.

    Example:
    documents/architecture/microservices.pdf
    category = architecture
    """
    if pdf_file.parent == DOCUMENT_DIR:
        return "general"

    return pdf_file.parent.name


def load_pdf_documents(pdf_file, file_hash):
    """
    Loads one PDF and adds metadata to each page.
    """
    loader = PyMuPDFLoader(str(pdf_file))
    documents = loader.load()

    category = get_pdf_category(pdf_file)

    for doc in documents:
        doc.metadata["source"] = pdf_file.name
        doc.metadata["file_path"] = str(pdf_file)
        doc.metadata["category"] = category
        doc.metadata["file_hash"] = file_hash

    return documents


def get_vector_store():
    """
    Opens existing ChromaDB vector store or creates it if missing.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def add_chunks_in_batches(vector_store, chunks):
    """
    Adds chunks to ChromaDB in batches.

    This avoids adding thousands of chunks in one large operation.
    """
    total_chunks = len(chunks)

    for start in range(0, total_chunks, BATCH_SIZE):
        end = start + BATCH_SIZE
        batch = chunks[start:end]

        print(f"Adding chunks {start + 1} to {min(end, total_chunks)} of {total_chunks}")

        vector_store.add_documents(batch)


def ingest_documents_v1_simple():
    """
    Version 1: Simple ingestion.

    Good for:
    - Small proof of concept
    - 2 or 3 PDFs
    - First prototype

    Limitation:
    - Reprocesses all files every time
    - Can create duplicates
    - Not ideal for large document sets
    """
    pdf_files = list(DOCUMENT_DIR.rglob("*.pdf"))

    all_documents = []

    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file.name}")

        file_hash = get_file_hash(pdf_file)
        documents = load_pdf_documents(pdf_file, file_hash)
        all_documents.extend(documents)

    chunks = split_documents(all_documents)

    vector_store = get_vector_store()
    add_chunks_in_batches(vector_store, chunks)

    print("V1 ingestion completed.")


def ingest_documents_incremental():
    """
    Version 3: Incremental ingestion with stale chunk deletion.

    Behavior:
    - New file: index it
    - Unchanged file: skip it
    - Changed file: delete old chunks, then re-index
    """
    manifest = load_manifest()
    pdf_files = list(DOCUMENT_DIR.rglob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    vector_store = get_vector_store()

    indexed_count = 0
    skipped_count = 0
    updated_count = 0

    for pdf_file in pdf_files:
        relative_path = str(pdf_file.relative_to(DOCUMENT_DIR))
        new_file_hash = get_file_hash(pdf_file)

        existing_entry = manifest.get(relative_path)

        if existing_entry and existing_entry.get("file_hash") == new_file_hash:
            print(f"Skipping unchanged file: {relative_path}")
            skipped_count += 1
            continue

        # Parse and validate new chunks before removing an existing document.
        documents = load_pdf_documents(pdf_file, new_file_hash)
        chunks = split_documents(documents)
        if not chunks:
            raise ValueError(f"No text chunks extracted from {relative_path}; index entry not updated.")

        if existing_entry and existing_entry.get("file_hash") != new_file_hash:
            print(f"Detected changed file: {relative_path}")

            updated_count += 1
        else:
            print(f"Detected new file: {relative_path}")

        print(f"Indexing file: {relative_path}")

        # Clear this path on retries as well as updates, so a partial failed
        # batch cannot accumulate duplicates. Do not delete another file that
        # happens to have identical content (and therefore the same hash).
        if existing_entry:
            del manifest[relative_path]
            save_manifest(manifest)
        vector_store._collection.delete(where={"file_path": str(pdf_file)})
        add_chunks_in_batches(vector_store, chunks)

        manifest[relative_path] = {
            "file_hash": new_file_hash,
            "source": pdf_file.name,
            "category": get_pdf_category(pdf_file),
            "chunk_count": len(chunks),
            "index_config": INDEX_CONFIG,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

        save_manifest(manifest)

        indexed_count += 1

    print("\nIncremental ingestion completed.")
    print(f"Indexed new/changed files: {indexed_count}")
    print(f"Updated files: {updated_count}")
    print(f"Skipped unchanged files: {skipped_count}")


def ingest_documents():
    """
    Default ingestion entry point used by main.py.
    """
    ingest_documents_incremental()


if __name__ == "__main__":
    ingest_documents()
