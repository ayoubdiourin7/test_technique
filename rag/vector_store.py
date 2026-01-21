from __future__ import annotations

import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from rag.config import CHROMA_DIR, EMBEDDINGS_MODEL_NAME, OPENAI_API_KEY


@lru_cache(maxsize=1)
def init_embedder() -> OpenAIEmbeddings:
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY is missing; embeddings cannot be initialized.")
        raise ValueError("OPENAI_API_KEY is required to initialize embeddings.")
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDINGS_MODEL_NAME)


@lru_cache(maxsize=1)
def init_vector_store(name: str = "documents") -> Chroma:
    embedder = init_embedder()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=name,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedder,
    )


def add_chunks_to_store(
    vector_store: Chroma,
    *,
    chunks: list[str],
    doc_id: str,
    source_path: str,
    doc_format: str | None = None,
    original_name: str | None = None,
) -> list[str]:
    if not chunks:
        return []

    ids = [f"{doc_id}_chunk_{idx:04d}" for idx in range(len(chunks))]
    metadatas = []
    for idx in range(len(chunks)):
        metadata = {
            "doc_id": doc_id,
            "chunk_index": idx,
            "source_path": source_path,
            "chunk_id": ids[idx],
        }
        if doc_format:
            metadata["doc_format"] = doc_format
        if original_name:
            metadata["original_name"] = original_name
        metadatas.append(metadata)
    
    vector_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

    return ids


def delete_chunks_from_store(vector_store: Chroma, doc_ids: list[str]) -> bool:
    try:
        vector_store.delete(ids=doc_ids)
        return True
    except Exception:
        logging.exception("Failed to delete chunks", extra={"doc_ids": doc_ids})
        return False
