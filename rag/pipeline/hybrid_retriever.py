from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, List

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

from rag.config import HYBRID_K, LEXICAL_WEIGHT
from rag.vector_store import init_vector_store


@dataclass
class HybridRetriever:
    dense_k: int = HYBRID_K
    lexical_k: int = HYBRID_K
    lexical_weight: float = LEXICAL_WEIGHT

    # Shared BM25 cache across instances
    _bm25: ClassVar[BM25Retriever | None] = None
    _bm25_docs: ClassVar[List[Document]] = []
    _bm25_ready: ClassVar[bool] = False
    _bm25_stale: ClassVar[bool] = True

    def __post_init__(self) -> None:
        vector_store = init_vector_store()
        self._dense = vector_store.as_retriever(search_kwargs={"k": self.dense_k * 2})

    @classmethod
    def notify_docs_changed(cls) -> None:
        cls._bm25 = None
        cls._bm25_docs = []
        cls._bm25_ready = False
        cls._bm25_stale = True

    @classmethod
    def _rebuild_bm25_index(cls, lexical_k: int) -> None:
        cls._bm25_ready = False
        cls._bm25_stale = True
        try:
            vector_store = init_vector_store()
            data = vector_store.get(include=["documents", "metadatas"])
            texts = data.get("documents", []) or []
            metadatas = data.get("metadatas", []) or []
            ids = data.get("ids", []) or []

            cls._bm25_docs = []
            if not texts:
                cls._bm25 = None
                cls._bm25_stale = False
                return

            for idx, text in enumerate(texts):
                meta = (metadatas[idx] if idx < len(metadatas) else {}) or {}
                id_ = ids[idx] if idx < len(ids) else None
                chunk_id = meta.get("chunk_id") or f"{meta.get('doc_id')}::{meta.get('chunk_index')}"
                if not chunk_id:
                    chunk_id = id_ or text[:50]
                merged_meta = {**meta, "chunk_id": chunk_id}
                cls._bm25_docs.append(Document(page_content=text, metadata=merged_meta))

            cls._bm25 = BM25Retriever.from_documents(cls._bm25_docs)
            cls._bm25.k = lexical_k * 2
            cls._bm25_ready = True
            cls._bm25_stale = False
        except Exception:
            logging.exception("Failed to rebuild BM25 index; lexical search disabled.")
            cls._bm25_docs = []
            cls._bm25 = None
            cls._bm25_ready = False
            cls._bm25_stale = False

    def _ensure_bm25(self) -> None:
        if (
            self.__class__._bm25_stale
            or not self.__class__._bm25_ready
            or not self.__class__._bm25_docs
            or not self.__class__._bm25
        ):
            self.__class__._rebuild_bm25_index(self.lexical_k)
        elif self.__class__._bm25:
            # Keep k in sync with the latest lexical_k
            self.__class__._bm25.k = self.lexical_k * 2

    def _fuse(self, dense_docs: List[Document], lexical_docs: List[Document], k: int) -> List[Document]:
        # Reciprocal Rank Fusion with optional lexical weighting.
        scores = defaultdict(float)
        seen_doc: dict[str, Document] = {}

        def key(doc: Document) -> str:
            meta = doc.metadata or {}
            return (
                meta.get("chunk_id")
                or f"{meta.get('doc_id')}::{meta.get('chunk_index')}"
                or meta.get("source_path")
                or doc.page_content[:50]
            )

        for rank, doc in enumerate(dense_docs, start=1):
            doc_key = key(doc)
            seen_doc[doc_key] = doc
            scores[doc_key] += 1 / (rank + 60.0)

        weight = max(min(self.lexical_weight, 1.0), 0.0)
        for rank, doc in enumerate(lexical_docs, start=1):
            doc_key = key(doc)
            if doc_key not in seen_doc:
                seen_doc[doc_key] = doc
            scores[doc_key] += weight * (1 / (rank + 60.0))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        fused: List[Document] = []
        for doc_key, _score in ranked:
            fused.append(seen_doc[doc_key])
            if len(fused) >= k:
                break
        return fused

    @staticmethod
    def _normalize_docs(raw_docs: List[Document | str]) -> List[Document]:
        normalized: List[Document] = []
        for item in raw_docs:
            if isinstance(item, Document):
                normalized.append(item)
            else:
                text = str(item)
                normalized.append(Document(page_content=text, metadata={"chunk_id": text[:50]}))
        return normalized

    def invoke(self, query: str, *, k: int) -> List[Document]:
        dense_docs: List[Document] = []
        lexical_docs: List[Document] = []

        try:
            dense_docs = self._dense.invoke(query) or []
        except Exception:
            logging.exception("Dense retrieval failed.")

        # Support tests or callers that explicitly disable lexical retrieval.
        if "_bm25_ready" in self.__dict__ and self.__dict__["_bm25_ready"] is False:
            return dense_docs[:k]

        self._ensure_bm25()
        bm25 = self.__class__._bm25
        if bm25 and self.__class__._bm25_docs:
            try:
                bm25.k = self.lexical_k * 2
                lexical_docs = bm25.invoke(query) or []
            except Exception:
                logging.exception("Lexical retrieval failed; continuing with dense only.")
                lexical_docs = []

        lexical_docs = self._normalize_docs(lexical_docs)

        if dense_docs and lexical_docs:
            dense_docs_normalized = self._normalize_docs(dense_docs)
            return self._fuse(dense_docs_normalized, lexical_docs, k)
        if dense_docs:
            return dense_docs[:k]
        if lexical_docs:
            return lexical_docs[:k]
        return []
