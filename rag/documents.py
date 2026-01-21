from __future__ import annotations

import logging
from langchain_chroma import Chroma
from pathlib import Path
from uuid import uuid4

from rag.chunking import chunk_text
from rag.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    REGISTRY_DB_PATH,
    UPLOADS_DIR,
    USE_TIKTOKEN,
)
from rag.pipeline.hybrid_retriever import HybridRetriever
from rag.preprocessing import preprocess_file
from rag.registry import DocumentRecord, DocumentRegistry
from rag.vector_store import (
    add_chunks_to_store,
    delete_chunks_from_store,
    init_vector_store,
)

# Initialize persistent store and registry singletons
vector_store: Chroma = init_vector_store()
registry = DocumentRegistry(REGISTRY_DB_PATH)



def _safe_name(name: str) -> str:
    return Path(name).name.replace(" ", "_")


def ingest_upload(filename: str, data: bytes) -> tuple[DocumentRecord, int]:
    """
    Ingest an uploaded file: store it, preprocess, chunk, embed, and register.
    """
    doc_id = uuid4().hex
    original_name = _safe_name(filename)
    stored_name = f"{doc_id}_{original_name}"
    stored_path = UPLOADS_DIR / stored_name
    stored_path.write_bytes(data)
    ext = stored_path.suffix.lower().lstrip(".")

    text = preprocess_file(stored_path)
    chunks = chunk_text(
        text,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
        use_tiktoken=USE_TIKTOKEN,
    )

    chunk_ids = add_chunks_to_store(
        vector_store,
        chunks=chunks,
        doc_id=doc_id,
        source_path=str(stored_path),
        doc_format=ext,
        original_name=original_name,
    )

    record = DocumentRecord(
        doc_id=doc_id,
        original_name=original_name,
        stored_path=str(stored_path),
        ext=ext,
        chunk_ids=chunk_ids,
    )
    registry.add(record)
    HybridRetriever.notify_docs_changed()
    return record, len(chunks)


def list_documents() -> list[DocumentRecord]:
    return registry.list()


def delete_document(doc_id: str) -> bool:
    record = registry.get(doc_id)
    if not record:
        return False

    if record.chunk_ids:
        delete_chunks_from_store(vector_store, record.chunk_ids)
    else:
        vector_store.delete(where={"doc_id": doc_id})

    stored_path = Path(record.stored_path)
    if stored_path.exists():
        stored_path.unlink()

    registry.remove(doc_id)
    HybridRetriever.notify_docs_changed()
    return True


def reset_document_store() -> None:
    """
    Delete all indexed documents, their vectors, and registry entries.
    Safe wrapper that reuses delete_document for consistency.
    """
    for record in list_documents():
        try:
            delete_document(record.doc_id)
        except Exception:
            logging.exception("Failed to delete document during reset", extra={"doc_id": record.doc_id})
