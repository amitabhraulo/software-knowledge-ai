from pathlib import Path

from src.config import VECTOR_DB_DIR
#from src.ingest.ingest import ingest_documents
from src.ingest.enterprise_ingest import ingest_documents
from src.rag.rag import ask_question


def is_vector_db_ready():
    """
    Checks whether vector database already exists.
    If vector_db folder has files, we assume ingestion is already done.
    """
    return VECTOR_DB_DIR.exists() and any(VECTOR_DB_DIR.iterdir())


def initialize_knowledge_base():
    """
    If vector database is missing, build it automatically.
    If vector database already exists, skip ingestion.
    """

    if is_vector_db_ready():
        print("Knowledge base already exists. Skipping ingestion.")
        return

    print("Knowledge base not found.")
    print("Building vector database from PDF documents...")

    ingest_documents()

    print("Knowledge base created successfully.")


def main():
    """
    Main application entry point.

    User only runs this file.
    """

    initialize_knowledge_base()

    print("\nSoftware Knowledge AI")
    print("LangChain RAG over software architecture PDFs.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nQuestion: ")

        if question.lower() in ["exit", "quit"]:
            print("Thanks Amitabh, Goodbye!")
            break

        answer = ask_question(question)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()