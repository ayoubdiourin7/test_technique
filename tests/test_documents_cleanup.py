from pathlib import Path

from rag import documents
from rag.registry import DocumentRecord


class _StubRegistry:
    def __init__(self, record: DocumentRecord):
        self.records = {record.doc_id: record}

    def get(self, doc_id: str):
        return self.records.get(doc_id)

    def remove(self, doc_id: str):
        self.records.pop(doc_id, None)

    def list(self):
        return list(self.records.values())


class _StubVectorStore:
    def __init__(self):
        self.deleted = []

    def delete(self, ids=None, where=None):
        if ids:
            self.deleted.extend(ids)
        elif where:
            self.deleted.append(where)
        return True


def test_delete_document_removes_store_and_file(tmp_path, monkeypatch):
    stored_path = tmp_path / "doc.txt"
    stored_path.write_text("hello")
    record = DocumentRecord(
        doc_id="doc123",
        original_name="doc.txt",
        stored_path=str(stored_path),
        ext="txt",
        chunk_ids=["c1", "c2"],
    )

    registry = _StubRegistry(record)
    vector_store = _StubVectorStore()

    monkeypatch.setattr(documents, "registry", registry)
    monkeypatch.setattr(documents, "vector_store", vector_store)

    success = documents.delete_document(record.doc_id)

    assert success is True
    assert vector_store.deleted == ["c1", "c2"]
    assert not stored_path.exists()
    assert record.doc_id not in registry.records


def test_reset_document_store_deletes_all(monkeypatch):
    records = [
        DocumentRecord(doc_id="a", original_name="a.txt", stored_path="a", ext="txt", chunk_ids=[]),
        DocumentRecord(doc_id="b", original_name="b.txt", stored_path="b", ext="txt", chunk_ids=[]),
    ]
    deleted_ids = []

    def fake_list():
        return records

    def fake_delete(doc_id):
        deleted_ids.append(doc_id)
        return True

    monkeypatch.setattr(documents, "list_documents", fake_list)
    monkeypatch.setattr(documents, "delete_document", fake_delete)

    documents.reset_document_store()

    assert deleted_ids == ["a", "b"]
