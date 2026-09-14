# Software Knowledge AI

> Enterprise Retrieval-Augmented Generation (RAG) for Software Architecture Knowledge

## Vision

Large engineering organizations maintain thousands of architecture documents, DevOps runbooks, cloud reference architectures, design decisions and operational guides.

Traditional keyword search returns documents containing keywords but often fails to return the document that answers the user's question.

This project demonstrates an enterprise-oriented Retrieval-Augmented Generation (RAG) architecture that transforms technical documents into a searchable knowledge base.

---

# Enterprise Problem

Traditional Search

Engineer → Search → 100 PDFs → Manual Reading

Enterprise AI

Engineer → Semantic Search → Relevant Context → LLM → Grounded Answer

---

# High Level Architecture

```
PDF Documents
      |
PyMuPDF Loader
      |
Recursive Chunking
      |
HuggingFace Embeddings
      |
Chroma Vector Database
      |
LangChain Retriever
      |
Prompt Construction
      |
Groq LLM
      |
Answer
```

---

# Project Structure

```text
software-knowledge-ai/
├── documents/
├── vector_db/
├── src/
│   ├── config.py
│   ├── main.py
│   ├── ingest/
│   │     ├── ingest.py
│   │     └── enterprise_ingest.py
│   ├── search/
│   │     └── retriever.py
│   ├── rag/
│   │     └── rag.py
│   └── llm/
│         └── groq_llm.py
└── README.md
```

---

# Version Evolution

## V1

- Load every PDF
- Chunk everything
- Generate embeddings
- Store into ChromaDB

Limitations

- Re-index every execution
- Slow for large repositories

## V2

- File hashing
- Manifest
- Incremental indexing
- Batch insertion

Benefits

- Skip unchanged files
- Faster ingestion

## V3

- Detect modified files
- Remove stale chunks
- Re-index only updated documents

Benefits

- Prevent duplicate vectors
- Keep vector database consistent

---

# Runtime Flow

```
main.py
   |
Knowledge Base Initialization
   |
Incremental Ingestion
   |
User Question
   |
Retriever (Top-K)
   |
Context Builder
   |
Groq LLM
   |
Answer
```

---

# Technology Choices

| Technology | Purpose |
|------------|---------|
| LangChain | AI orchestration |
| HuggingFace | Embeddings |
| ChromaDB | Vector storage |
| Groq | LLM inference |
| PyMuPDF | PDF loading |

---

# Architectural Trade-offs

- Chunk size vs context quality
- Top-K vs token cost
- Local embeddings vs hosted APIs
- ChromaDB vs distributed vector databases
- Full indexing vs incremental indexing

---

# Future Roadmap

- Hybrid Search
- Metadata filtering
- Cross-Encoder reranking
- Chunk versioning
- Page-level hashing
- Multi-agent ingestion
- Qdrant / Milvus support
- Evaluation framework

---

# Sample Questions

- Explain CQRS.
- Explain Saga Pattern.
- What is Database per Service?
- Explain API Gateway.
- Explain communication between microservices.

