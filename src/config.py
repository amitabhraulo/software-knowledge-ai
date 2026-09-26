import hashlib
import json
from pathlib import Path

# Root folder of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder where PDFs are stored
DOCUMENT_DIR = BASE_DIR / "documents"

# Folder where ChromaDB will store vectors
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Hugging Face embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Token budgets for this model, not character counts. Review the limit when
# changing models; reserve space for the tokenizer's special tokens.
EMBEDDING_MAX_TOKENS = 256
CHUNK_SIZE = 240
CHUNK_OVERLAP = 40

INDEX_CONFIG = {
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_max_tokens": EMBEDDING_MAX_TOKENS,
    "splitter": "recursive_embedding_tokens_v1",
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
}
# Incompatible settings get their own collection and manifest. Old indexes
# remain intact, and ingestion/retrieval always select the same configuration.
INDEX_ID = hashlib.sha256(json.dumps(INDEX_CONFIG, sort_keys=True).encode()).hexdigest()[:16]
COLLECTION_NAME = f"software_knowledge_base_{INDEX_ID}"
MANIFEST_FILE = VECTOR_DB_DIR / f"index_manifest_{INDEX_ID}.json"
BATCH_SIZE = 100
