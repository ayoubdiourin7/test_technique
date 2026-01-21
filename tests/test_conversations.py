from rag.conversations import ConversationStore


def test_ensure_default_creates_once(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")

    first = store.ensure_default_conversation()
    second = store.ensure_default_conversation()

    assert first.conversation_id == second.conversation_id
    assert store.count_conversations() == 1


def test_add_message_persists_and_updates_timestamp(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conv = store.create_conversation("Test")
    original_updated = conv.updated_at

    store.add_message(conv.conversation_id, "user", "Hello", sources=[])
    assistant_msg = store.add_message(
        conv.conversation_id,
        "assistant",
        "Hi there",
        sources=[{"doc_id": "d1", "original_name": "file.txt"}],
    )

    messages = store.list_messages(conv.conversation_id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[-1].content == "Hi there"
    assert messages[-1].sources == [{"doc_id": "d1", "original_name": "file.txt"}]

    refreshed = store.get_conversation(conv.conversation_id)
    assert refreshed is not None
    assert refreshed.updated_at != original_updated
    assert refreshed.updated_at >= assistant_msg.created_at


def test_delete_conversation_removes_messages(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conv = store.create_conversation("To delete")
    store.add_message(conv.conversation_id, "user", "Hello", sources=[])

    assert store.delete_conversation(conv.conversation_id) is True
    assert store.list_messages(conv.conversation_id) == []
    assert store.get_conversation(conv.conversation_id) is None
    assert store.delete_conversation("missing") is False
