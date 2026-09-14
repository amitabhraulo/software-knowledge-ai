"""
This file creates the Groq LLM object using LangChain.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env variables into environment.
load_dotenv()

def get_llm():
    """
    Creates and returns a Groq chat model.

    The API key is read automatically from GROQ_API_KEY in .env.
    The model name is read from GROQ_MODEL in .env.
    """

    return ChatGroq(
        model = os.getenv("GROQ_MODEL")
    )

if __name__ == "__main__":
    llm = get_llm()

    # Simple test without RAG.
    response = llm.invoke("Explain microservices in 5 simple lines.")

    print(response.content)