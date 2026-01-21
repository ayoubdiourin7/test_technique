from rag.registry import DocumentRecord, DocumentRegistry


def test_registry_add_get_list_remove(tmp_path) -> None:
    db_path = tmp_path / "registry.sqlite3"
    registry = DocumentRegistry(db_path)
    record = DocumentRecord(
        doc_id="doc-123",
        original_name="source.txt",
        stored_path="/tmp/source.txt",
        ext="txt",
        chunk_ids=["chunk-1", "chunk-2"],
    )

    registry.add(record)

    listed = registry.list()
    assert len(listed) == 1
    assert listed[0].doc_id == record.doc_id

    fetched = registry.get(record.doc_id)
    assert fetched == record

    removed = registry.remove(record.doc_id)
    assert removed == record
    assert registry.get(record.doc_id) is None
    assert registry.list() == []


def test_registry_remove_unknown_returns_none(tmp_path) -> None:
    db_path = tmp_path / "registry.sqlite3"
    registry = DocumentRegistry(db_path)

    assert registry.remove("missing-doc") is None
