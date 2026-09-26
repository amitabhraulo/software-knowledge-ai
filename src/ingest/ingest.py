"""
This file creates the vector database.

Flow:
PDF files -> Load pages -> Split into chunks -> Create embeddings -> Store in ChromaDB
"""

from langchain_community.document_loaders import PyMuPDFLoader
from src.ingest.chunking import split_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
    DOCUMENT_DIR,
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


def load_documents():
    """
    Loads all PDF documents from the documents folder.

    LangChain returns a list of Document objects.
    Each Document usually represents one PDF page.
    """

    all_documents=[]
    # Find all PDF files inside documents folder.
    pdf_files = list(DOCUMENT_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in documents folder.")
        return all_documents
    
    # Read every PDF file.
    for pdf_file in pdf_files:
        print(f"loading PDF: {pdf_file.name}")

        # PyPDFLoader reads the PDF and converts pages into LangChain Document objects.
        loader = PyMuPDFLoader(str(pdf_file))
        # Load PDF pages.
        documents = loader.load()

        # Add source filename to each page metadata.
        # This helps to show the source PDF later.
        for doc in documents:
            doc.metadata["source"] = pdf_file.name

        # Add this PDF's pages to the full list.
        all_documents.extend(documents)

    return all_documents

def ingest_documents():
    """
    Main ingestion pipeline.

    This function:
    1. Loads PDFs
    2. Splits pages into chunks
    3. Creates embeddings
    4. Saves everything into ChromaDB
    """

    # Step 1: Load all PDF pages.
    documents = load_documents()

    # Stop if no documents were loaded.
    if not documents:
        return
    
    chunks = split_documents(documents)
    print(f"Total pages loaded: {len(documents)}")

    # Step 3: Create Hugging Face embedding model.
    # This downloads the model on first run and then uses local cache.
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Step 4: Store chunks + embeddings into ChromaDB.
    # persist_directory ensures the vector DB is saved on disk.

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=COLLECTION_NAME
    )

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest_documents()
