from src.ingest.enterprise_ingest import ingest_documents
from src.rag.rag import ask_question


def initialize_knowledge_base():
    """Refresh the active index; incremental ingestion skips unchanged PDFs."""
    print("Checking PDFs and the active embedding/chunking configuration...")
    ingest_documents()


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
