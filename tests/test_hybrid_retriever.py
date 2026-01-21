from rag.pipeline.hybrid_retriever import HybridRetriever
from langchain.schema import Document


def test_hybrid_retriever_fallback_dense_only(monkeypatch):
    retriever = HybridRetriever(dense_k=2, lexical_k=2, lexical_weight=0.5)

    class DummyDense:
        def invoke(self, query):
            return ["dense1", "dense2"]

    retriever._dense = DummyDense()  # type: ignore
    retriever._bm25_ready = False
    docs = retriever.invoke("query", k=2)

    assert docs == ["dense1", "dense2"]


def test_hybrid_retriever_keeps_distinct_chunks_and_dedups_same_chunk():
    retriever = HybridRetriever(dense_k=3, lexical_k=3, lexical_weight=0.5)
    doc_a1 = Document(page_content="A1", metadata={"chunk_id": "docA::0"})
    doc_a1_dup = Document(page_content="A1", metadata={"chunk_id": "docA::0"})
    doc_a2 = Document(page_content="A2", metadata={"chunk_id": "docA::1"})

    fused = retriever._fuse([doc_a1, doc_a2], [doc_a1_dup], k=5)

    assert len(fused) == 2
    assert any(d.metadata["chunk_id"] == "docA::0" for d in fused)
    assert any(d.metadata["chunk_id"] == "docA::1" for d in fused)


def test_notify_docs_changed_marks_bm25_stale():
    HybridRetriever.notify_docs_changed()
    assert HybridRetriever._bm25_stale is True
    assert HybridRetriever._bm25_ready is False
