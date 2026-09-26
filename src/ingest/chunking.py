"""Preserve text boundaries while measuring chunks with the embedding tokenizer."""

from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from src.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL_NAME,
)


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def split_documents(documents):
    tokenizer = get_tokenizer()
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE <= EMBEDDING_MAX_TOKENS - special_tokens:
        raise ValueError("Chunk size/overlap must fit the embedding model, including special tokens.")

    def count_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))

    # Prefer paragraph/newline/word boundaries, then characters as a fallback.
    # Overlap is a target: natural boundaries can result in less overlap.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=count_tokens,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        token_count = count_tokens(chunk.page_content)
        # Tokenization is not strictly additive across joined pieces. Check
        # final chunks too, so no oversized text is silently embedded partially.
        if token_count > CHUNK_SIZE:
            raise ValueError("Splitter produced an oversized chunk; reduce the chunk budget.")
        chunk.metadata["token_count"] = token_count
    return chunks
