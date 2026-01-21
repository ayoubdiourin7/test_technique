from types import SimpleNamespace

from langchain.schema import Document
from langchain_core.runnables import RunnableLambda

from rag.pipeline import qa


class _DummyRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, _query, *_args, **_kwargs):
        return self._docs


def test_answer_question_no_docs_returns_message(monkeypatch):
    monkeypatch.setattr(
        qa,
        "HybridRetriever",
        lambda dense_k, lexical_k: _DummyRetriever([]),
    )
    dummy_llm = RunnableLambda(lambda _msgs: SimpleNamespace(content="unused"))

    answer, sources = qa.answer_question("Question ?", top_k=1, history=[], llm=dummy_llm)

    assert "aucun document" in answer.lower()
    assert sources == []


def test_answer_question_requires_citations(monkeypatch):
    docs = [Document(page_content="Context", metadata={"doc_id": "doc1", "chunk_id": "doc1::0"})]
    monkeypatch.setattr(
        qa,
        "HybridRetriever",
        lambda dense_k, lexical_k: _DummyRetriever(docs),
    )
    dummy_llm = RunnableLambda(lambda _msgs: "Réponse sans citation")

    answer, sources = qa.answer_question("Question ?", top_k=1, history=[], llm=dummy_llm)

    assert "citation" in answer.lower()
    assert sources == []
