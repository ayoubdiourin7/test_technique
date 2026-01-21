import logging
import re
from pathlib import Path

import streamlit as st

from rag.conversations import get_conversation_store
from rag.documents import list_documents
from rag.pipeline import answer_question, sanitize_question

logger = logging.getLogger(__name__)

st.title("💬 Chat")
st.info("Posez vos questions sur les documents indexés. Réponses RAG uniquement.")

# Hide the default home page entry so only Chat and Documents are visible.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] ul li:first-child {display: none;}
    /* Chat-style bubbles */
    [data-testid="stChatMessage"] {
        padding: 0.35rem 0;
    }
    [data-testid="stChatMessage"] > div {
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stChatMessage"][data-testid="stChatMessage-user"] > div {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
    }
    [data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] > div {
        background: #eef2ff;
        border: 1px solid #e0e7ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

store = get_conversation_store()
if "conversation_id" not in st.session_state:
    default_conv = store.ensure_default_conversation()
    st.session_state["conversation_id"] = default_conv.conversation_id


def _render_sources(sources: list[dict], doc_names: dict[str, str]) -> None:
    if not sources:
        return
    st.markdown("**Sources utilisées**")
    for src in _dedup_sources(sources):
        path = src.get("source_path")
        label = (
            src.get("original_name")
            or doc_names.get(src.get("doc_id", ""), None)
            or (Path(path).name if path else "inconnu")
        )
        st.markdown(f"- `{label}`")


def _dedup_sources(sources: list[dict]) -> list[dict]:
    seen = set()
    deduped: list[dict] = []
    for src in sources:
        key = src.get("doc_id") or src.get("source_path") or src.get("original_name")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(src)
    return deduped


def _select_cited_sources(answer: str, sources: list[dict]) -> list[dict]:
    if not sources:
        return []
    matches = re.findall(r"\[(\d+)\]", answer)
    indices: list[int] = []
    for match in matches:
        try:
            idx = int(match) - 1
        except ValueError:
            continue
        if 0 <= idx < len(sources) and idx not in indices:
            indices.append(idx)
    if not indices:
        return _dedup_sources(sources)
    selected = []
    seen_keys = set()
    for idx in indices:
        src = sources[idx]
        key = src.get("doc_id") or src.get("source_path") or src.get("original_name")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(src)
    return selected


# Sidebar: ChatGPT-style conversation list + new chat
with st.sidebar:
    st.subheader("Conversations")
    if st.button("➕ Nouvelle conversation", use_container_width=True):
        new_conv = store.create_conversation()
        st.session_state["conversation_id"] = new_conv.conversation_id
        st.rerun()

    conversations = store.list_conversations()
    if not conversations:
        default_conv = store.ensure_default_conversation()
        conversations = [default_conv]
        st.session_state["conversation_id"] = default_conv.conversation_id

    options = [conv.conversation_id for conv in conversations]
    current_id = st.session_state.get("conversation_id", options[0])
    if current_id not in options:
        current_id = options[0]

    selected_id = st.radio(
        "Historique",
        options=options,
        index=options.index(current_id),
        format_func=lambda cid: next(
            (f"{c.title} · {c.updated_at.split('T')[0]}" for c in conversations if c.conversation_id == cid),
            "Conversation",
        ),
    )
    if selected_id != st.session_state["conversation_id"]:
        st.session_state["conversation_id"] = selected_id
        st.rerun()

    if st.button("🗑️ Supprimer la conversation", use_container_width=True, disabled=not conversations):
        store.delete_conversation(selected_id)
        next_conv = store.get_most_recent() or store.ensure_default_conversation()
        st.session_state["conversation_id"] = next_conv.conversation_id
        st.rerun()

conversation_id = st.session_state["conversation_id"]

# Conversation header
active_conv = store.get_conversation(conversation_id)
st.subheader(active_conv.title if active_conv else "Conversation")
if active_conv:
    st.caption(f"Dernière mise à jour : {active_conv.updated_at}")

documents = list_documents()
doc_names_map = {doc.doc_id: doc.original_name for doc in documents}
no_docs = len(documents) == 0
if no_docs:
    st.warning("Aucun document indexé. Ajoutez-en via l’onglet Documents.")

# Render history
messages = store.list_messages(conversation_id)
for message in messages:
    with st.chat_message(message.role):
        st.markdown(message.content)
        if message.role == "assistant":
            _render_sources(message.sources or [], doc_names_map)

history_payload = [{"role": m.role, "content": m.content} for m in messages]
question = st.chat_input("Posez une question", disabled=no_docs)

if question:
    sanitized_question, refusal_reason = sanitize_question(
        question, raise_on_refusal=False
    )
    if refusal_reason or not sanitized_question:
        logger.warning("User input refused", extra={"reason": refusal_reason})
        with st.chat_message("assistant"):
            st.warning(refusal_reason or "La requête est invalide. Aucun message enregistré.")
        st.stop()

    user_msg = store.add_message(
        conversation_id, role="user", content=sanitized_question, sources=[]
    )
    st.chat_message("user").markdown(sanitized_question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des passages..."):
            try:
                answer, sources = answer_question(
                    sanitized_question,
                    history=history_payload,
                )
            except ValueError as err:
                logger.warning("Handled ValueError during QA", exc_info=True)
                st.error(str(err))
                st.stop()
            except Exception as err:
                logger.exception("Unexpected error during QA")
                st.error("Une erreur est survenue lors de la génération de la réponse.")
                st.exception(err)
                st.stop()

        st.markdown(answer)
        cited = _select_cited_sources(answer, sources)
        _render_sources(cited, doc_names_map)

    assistant_msg = store.add_message(
        conversation_id, role="assistant", content=answer, sources=cited
    )
    st.rerun()
