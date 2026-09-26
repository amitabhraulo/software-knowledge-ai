"""Scored retrieval from the existing index; no metric change or re-indexing."""

from dataclasses import dataclass
from functools import lru_cache
import math

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, VECTOR_DB_DIR


@dataclass
class Hit:
    document: Document
    distance: float
    accepted: bool


@dataclass
class Retrieval:
    hits: list[Hit]
    metric: str
    filters: dict | None
    max_distance: float | None

    @property
    def documents(self):
        return [hit.document for hit in self.hits if hit.accepted]


@lru_cache(maxsize=1)
def get_search_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
        create_collection_if_not_exists=False,
    )


def build_filter(source=None, category=None):
    conditions = []
    for key, value in (("source", source), ("category", category)):
        if value is not None:
            if not value.strip():
                raise ValueError(f"{key} cannot be blank")
            conditions.append({key: value})
    if len(conditions) > 1:
        return {"$and": conditions}
    return conditions[0] if conditions else None


def retrieve_scored(question, *, top_k=5, source=None, category=None,
                    max_distance=None, store=None):
    if not question.strip():
        raise ValueError("Question cannot be blank")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if max_distance is not None and not math.isfinite(max_distance):
        raise ValueError("max_distance must be finite")
    filters = build_filter(source, category)
    if store is None:
        store = get_search_store()
    # Read the actual collection settings rather than guessing the score scale.
    config = store._collection.configuration
    index = config.get("hnsw") or config.get("spann") or {}
    metric = index.get("space")
    if metric not in {"l2", "cosine", "ip"}:
        raise ValueError(f"Unsupported or unknown collection metric: {metric!r}")
    pairs = store.similarity_search_with_score(question, k=top_k, filter=filters)
    hits = []
    for document, distance in pairs:
        distance = float(distance)
        if not math.isfinite(distance):
            raise ValueError("Vector store returned a non-finite distance")
        hits.append(Hit(document, distance,
                        max_distance is None or distance <= max_distance))
    return Retrieval(hits, metric, filters, max_distance)
