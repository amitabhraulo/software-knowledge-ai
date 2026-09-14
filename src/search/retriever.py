"""
This file performs semantic search.

Flow:
User question -> Embedding -> ChromaDB search -> Relevant document chunks
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import(
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)

def get_retriever(top_k=5):
    """
    Creates a retriever from the existing ChromaDB vector database.

    top_k means:
    Return the top K most relevant chunks.
    """

    # Load the same embedding model used during ingestion.
    embeddings= HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Connect to existing ChromaDB stored in vector_db folder.
    vector_store = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    # Convert vector store into LangChain retriever.
    retriever = vector_store.as_retriever(search_kwargs = {"k":top_k})

    return retriever

if __name__ == "__main__":
    retriever = get_retriever(top_k=5)

    question = input("Ask your question: ")

    # Retrieve matching chunks.
    docs = retriever.invoke(question)

    print(f"\nRetrieved {len(docs)} documents")

    for index, doc in enumerate(docs, start=1):
        print("\n----------------------")
        print(f"Result {index}")

        # Metadata usually contains source and page.
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))

        print("\nContent Preview:\n")
        print(doc.page_content[:1000])