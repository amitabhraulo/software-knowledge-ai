# Software Knowledge AI

> Enterprise Retrieval-Augmented Generation (RAG) for Software Architecture Knowledge

Software Knowledge AI is a document-based RAG application that transforms software architecture PDFs into a searchable knowledge base.

Instead of manually searching through multiple technical documents, an engineer can ask a question in natural language. The application retrieves relevant information from the indexed documents and uses an LLM to generate a grounded answer.

---

## Vision

Large engineering organizations maintain architecture documents, DevOps runbooks, cloud reference architectures, design decisions, operational guides, and technical documentation.

Traditional keyword search can locate documents containing specific words, but engineers still need to open multiple documents and manually identify the relevant information.

This project explores a different approach:

```text
Traditional Search

Engineer
   ↓
Keyword Search
   ↓
Multiple Documents
   ↓
Manual Reading
```

With Retrieval-Augmented Generation:

```text
Enterprise Knowledge AI

Engineer Question
      ↓
Semantic Search
      ↓
Relevant Document Context
      ↓
LLM
      ↓
Grounded Answer
```

The goal of this project is not just to build a chatbot, but to explore the architecture and engineering decisions involved in building an enterprise-oriented RAG system.

---

# High-Level Architecture

```text
PDF Documents
      ↓
PyMuPDF Loader
      ↓
Recursive Text Chunking
      ↓
HuggingFace Embeddings
      ↓
Chroma Vector Database
      ↓
LangChain Retriever
      ↓
Top-K Relevant Chunks
      ↓
Prompt Construction
      ↓
Groq LLM
      ↓
Grounded Answer
```

The architecture has two main flows:

### Document Ingestion

```text
PDF
 ↓
Load
 ↓
Chunk
 ↓
Generate Embeddings
 ↓
Store in ChromaDB
```

### Question Answering

```text
User Question
      ↓
Question Embedding
      ↓
Semantic Search
      ↓
Relevant Chunks
      ↓
Context
      ↓
Groq LLM
      ↓
Answer
```

HuggingFace embeddings are used for **semantic retrieval**, while Groq provides the **LLM inference** used to generate the final answer.

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| LangChain | RAG orchestration |
| HuggingFace | Local embedding generation |
| ChromaDB | Vector storage and similarity search |
| Groq | LLM inference |
| PyMuPDF | PDF document loading |
| uv | Python dependency and environment management |

---

# Project Structure

```text
software-knowledge-ai/
│
├── documents/
│   └── PDF documents used as the knowledge source
│
├── src/
│   ├── config.py
│   │
│   ├── ingest/
│   │   ├── ingest.py
│   │   └── enterprise_ingest.py
│   │
│   ├── search/
│   │   └── retriever.py
│   │
│   ├── rag/
│   │   └── rag.py
│   │
│   └── llm/
│       └── groq_llm.py
│
├── main.py
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

The following directories/files are generated locally and are intentionally not committed to GitHub:

```text
.env
.venv/
vector_db/
__pycache__/
```

---

# Prerequisites

Before running the project, install:

- Git
- Python
- uv
- A Groq API key

Check that Git is available:

```bash
git --version
```

Check Python:

```bash
python --version
```

Check uv:

```bash
uv --version
```

---

# Setting Up Groq for LLM Inference

## What is Groq?

Groq provides high-speed LLM inference and access to supported language models through an API.

This project uses Groq for generating answers after relevant context has been retrieved from the vector database.

The embedding model itself runs separately using HuggingFace.

---

## Create a Groq API Key

1. Open the Groq API Keys page:

   https://console.groq.com/keys

2. Sign in or create a Groq account.

3. Click **Create API Key**.

4. Enter a name for the key, for example:

   ```text
   software-knowledge-ai
   ```

5. Create the key and copy it.

6. Keep the API key secure. Do not commit it to GitHub.

---

# Clone and Run the Application

## 1. Clone the Repository

```bash
git clone https://github.com/amitabhraulo/software-knowledge-ai.git
```

Move into the project directory:

```bash
cd software-knowledge-ai
```

---

## 2. Install Dependencies

The project uses `uv` for Python dependency management.

Run:

```bash
uv sync
```

This creates a local virtual environment and installs the dependencies defined in:

```text
pyproject.toml
uv.lock
```

The generated `.venv/` directory is ignored by Git.

---

## 3. Configure Environment Variables

The repository contains:

```text
.env.example
```

Create a local `.env` file from it.

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### macOS / Linux

```bash
cp .env.example .env
```

Open `.env` and configure your Groq settings:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=your_groq_model_here
```

Replace the placeholder values with your actual Groq configuration.

> Never commit the `.env` file or your API key to GitHub.

---

## 4. Add Knowledge Documents

Place the PDF documents you want to query inside:

```text
documents/
```

For example:

```text
documents/
├── microservices-architecture.pdf
├── cloud-architecture.pdf
└── software-design-guide.pdf
```

These documents become the knowledge source for the RAG application.

---

## 5. Run the Application

Start the application using:

```bash
uv run python main.py
```

On the first run, the application builds the knowledge base.

```text
PDF Documents
      ↓
Document Loading
      ↓
Chunking
      ↓
Embedding Generation
      ↓
ChromaDB
      ↓
vector_db/
```

The `vector_db/` directory is generated locally and is not committed to GitHub.

After initialization, the application waits for a question:

```text
Software Knowledge AI
LangChain RAG over software architecture PDFs.

Question:
```

You can now ask questions such as:

```text
What is an API Gateway?
```

or:

```text
Explain the Saga Pattern.
```

The application retrieves relevant document chunks and sends the retrieved context to the Groq LLM to generate the answer.

Type:

```text
exit
```

or:

```text
quit
```

to stop the application.

---

# Runtime Flow

```text
main.py
   ↓
Knowledge Base Initialization
   ↓
Document Ingestion
   ↓
User Question
   ↓
Retriever
   ↓
Question Embedding
   ↓
ChromaDB Similarity Search
   ↓
Top-K Relevant Chunks
   ↓
Context Builder
   ↓
Prompt
   ↓
Groq LLM
   ↓
Grounded Answer
```

---

# Incremental Document Ingestion

Rebuilding embeddings for every document whenever the application changes would become expensive as the knowledge base grows.

The ingestion pipeline therefore evolved incrementally.

## V1 — Basic Ingestion

The initial implementation:

```text
Load every PDF
      ↓
Chunk every document
      ↓
Generate all embeddings
      ↓
Store everything in ChromaDB
```

### Limitations

- Re-indexes all documents
- Recreates embeddings unnecessarily
- Becomes inefficient as the document repository grows

---

## V2 — Incremental Indexing

The next version introduced:

- SHA-256 file hashing
- Index manifest
- Incremental document detection
- Batch insertion

For each document:

```text
PDF
 ↓
Calculate Hash
 ↓
Compare with Manifest
 ↓
Unchanged?
 ├── Yes → Skip
 └── No  → Index
```

### Benefits

- Unchanged files are skipped
- Only new documents require embeddings
- Faster ingestion for growing document repositories

---

## V3 — Modified Document Detection

Incremental ingestion also needs to handle existing documents that have changed.

```text
Existing PDF
      ↓
Calculate Current Hash
      ↓
Compare with Previous Hash
      ↓
Changed?
 ├── No  → Skip
 └── Yes
       ↓
Delete Old Chunks
       ↓
Reprocess Document
       ↓
Generate New Embeddings
       ↓
Update Vector Database
```

### Benefits

- Prevents duplicate vectors
- Removes stale document chunks
- Re-indexes only modified documents
- Keeps the vector database consistent with the source documents

---

# Manifest

The ingestion pipeline maintains an index manifest containing information about processed documents.

Conceptually:

```json
{
  "architecture.pdf": {
    "file_hash": "...",
    "source": "architecture.pdf",
    "category": "general",
    "chunk_count": 100
  }
}
```

The manifest allows the ingestion pipeline to determine whether a document is new, unchanged, or modified.

---

# Retrieval and RAG Flow

When a user asks:

```text
What is the Saga Pattern?
```

the question is converted into an embedding using the same embedding model used during document ingestion.

```text
Question
   ↓
HuggingFace Embedding
   ↓
Vector Representation
   ↓
ChromaDB Similarity Search
   ↓
Top-K Relevant Document Chunks
```

The retrieved chunks are then combined into context:

```text
Question
   +
Retrieved Context
   ↓
Prompt
   ↓
Groq LLM
   ↓
Answer
```

The prompt instructs the model to answer using the retrieved context rather than relying only on the LLM's general knowledge.

If the requested information cannot be found in the indexed documents, the application is designed to respond accordingly.

---

# Sample Questions

Depending on the documents you index, example questions include:

```text
Explain CQRS.

Explain the Saga Pattern.

What is Database per Service?

What is an API Gateway?

How do microservices communicate with each other?

What are the benefits of microservices architecture?
```

You can also test grounding using a question unrelated to the indexed documents.

The expected behavior is for the application to indicate that the information could not be found in the knowledge base rather than inventing an answer.

---

# Architectural Trade-offs

The project explores several RAG architecture decisions:

### Chunk Size vs Context Quality

Smaller chunks can improve retrieval precision but may lose surrounding context.

Larger chunks preserve more context but may introduce irrelevant information.

### Top-K vs Context Size

Retrieving more chunks can increase recall but also increases the amount of context sent to the LLM.

### Local vs Hosted Embeddings

This project uses HuggingFace embeddings locally, avoiding an external embedding API dependency.

### ChromaDB vs Distributed Vector Databases

ChromaDB is suitable for local development and experimentation.

Larger enterprise deployments may require distributed vector databases depending on scale, availability, filtering, and operational requirements.

### Full vs Incremental Indexing

Full indexing is simple but inefficient as document repositories grow.

Incremental indexing reduces unnecessary embedding generation by processing only new or modified documents.

---

# Fresh Start / Rebuild the Knowledge Base

The vector database is generated locally.

To rebuild the knowledge base from scratch, stop the application and delete:

```text
vector_db/
```

Then run:

```bash
uv run python main.py
```

The application will regenerate embeddings and rebuild the ChromaDB database from the documents.

Python may also create directories such as:

```text
__pycache__/
```

These contain generated Python bytecode and can safely be ignored. They are excluded from Git using `.gitignore`.

---

# Security

Never commit secrets to source control.

The following file should remain local:

```text
.env
```

The repository provides:

```text
.env.example
```

only as a configuration template.

Generated runtime directories such as the following are also excluded:

```text
.venv/
vector_db/
__pycache__/
```

---

# Future Roadmap

Planned improvements include:

- Source citations in generated answers
- Metadata filtering
- Hybrid search
- Cross-encoder reranking
- Improved chunking strategies
- Query rewriting
- Retrieval evaluation
- RAG evaluation framework
- Caching
- Guardrails
- Observability
- Chunk versioning
- Page-level hashing
- Qdrant / Milvus support
- API layer
- Docker support

---

# Learning Goal

This project is being developed incrementally to explore how a basic RAG pipeline can evolve toward a more reliable enterprise knowledge architecture.

The focus is not only on using an LLM, but on understanding the engineering decisions around:

```text
Document Ingestion
        +
Chunking
        +
Embeddings
        +
Vector Search
        +
Retrieval
        +
Context Construction
        +
LLM Generation
        +
Reliability
```

The project will continue to evolve as additional retrieval, evaluation, observability, and production-oriented patterns are introduced.