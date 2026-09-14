from pathlib import Path

# Root folder of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder where PDFs are stored
DOCUMENT_DIR = BASE_DIR / "documents"

# Folder where ChromaDB will store vectors
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# ChromaDB collection name
COLLECTION_NAME = "software_knowledge_base"

# Hugging Face embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

MANIFEST_FILE = VECTOR_DB_DIR / "index_manifest.json"
BATCH_SIZE = 100