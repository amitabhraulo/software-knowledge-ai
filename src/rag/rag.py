"""
This file connects Retriever + Prompt + LLM.

Flow:
Question -> Retriever -> Context -> Prompt -> Groq LLM -> Answer
"""

from langchain_core.prompts import ChatPromptTemplate
from src.search.retriever import get_retriever
from src.llm.groq_llm import get_llm

def format_docs(docs):
    """
    Converts retrieved documents into a single context string.

    The LLM receives this context and answers based on it.
    """

    context_parts= []

    for doc in docs:
        context_parts.append(
            f"""
            Source: {doc.metadata.get("source")}
            Page: {doc.metadata.get("page")}
            Content:{doc.page_content}
            """
        )
    
    return "\n".join(context_parts)

def ask_question(question):
    """
    Full RAG flow.

    1. Retrieve relevant chunks
    2. Convert chunks into context
    3. Build prompt
    4. Send prompt to LLM
    5. Return answer
    """

    # Get semantic retriever.
    retriever = get_retriever(top_k=5)

    # Retrieve relevant chunks for the question.
    docs = retriever.invoke(question)

    return generate_baseline_answer(question, docs)


def generate_baseline_answer(question, docs, llm=None):
    """Original prompt and generation, shared with comparison mode."""
    context = format_docs(docs)
    
    # Prompt template keeps question and context separate.
    prompt = ChatPromptTemplate.from_template(
        """
        You are a senior software architecture assistant.

        Answer the question using only the context below.
        If the answer is not available in the context, say:
        "I could not find this in the indexed documents."

        Question:
        {question}

        Context:
        {context}
        """
    )

    # Create Groq LLM.
    if llm is None:
        llm = get_llm()

     # LCEL chain: prompt output goes into LLM.
    chain = prompt | llm

    # Execute the chain.
    response = chain.invoke({
        "question": question,
        "context": context
    })

    # Return only text content.
    return response.content

if __name__ == "__main__":
    question = input("Ask your software architecture question: ")

    answer = ask_question(question)

    print("\nAnswer:\n")
    print(answer)