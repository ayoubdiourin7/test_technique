import logging
from pathlib import Path
import streamlit as st

from rag.config import DOC_PREVIEW_CHARS
from rag.documents import delete_document, ingest_upload, list_documents, reset_document_store

logger = logging.getLogger(__name__)

st.title("📄 Documents")
st.info(
    "Ajoutez des documents texte que le système pourra utiliser pour répondre aux questions. "
    "Formats : .txt, .csv, .html."
)

# Hide the default home page entry so only Chat and Documents are visible.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] ul li:first-child {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.subheader("Indexer de nouveaux documents")
if "uploader_nonce" not in st.session_state:
    st.session_state["uploader_nonce"] = 0
if "upload_key" in st.session_state:
    del st.session_state["upload_key"]

uploaded = st.file_uploader(
    "Sélectionnez un ou plusieurs fichiers",
    type=["txt", "csv", "html", "htm"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_nonce']}",
)

if st.button("Indexer", use_container_width=True, disabled=not uploaded):
    for upload in uploaded or []:
        try:
            record, chunk_count = ingest_upload(upload.name, bytes(upload.getbuffer()))
            st.success(f"Ajouté : {record.original_name} ({chunk_count} chunks)")
        except Exception as exc:
            logger.exception("Failed to ingest upload", extra={"file": upload.name})
            st.error(f"Impossible d'indexer {upload.name}: {exc}")
    st.session_state["uploader_nonce"] += 1
    st.rerun()

st.divider()
st.subheader("Documents enregistrés")

documents = list_documents()
if not documents:
    st.info("Aucun document pour le moment.")
else:
    if st.button("🔄 Réinitialiser l'index", type="secondary"):
        try:
            reset_document_store()
            st.success("Index vidé.")
            st.rerun()
        except Exception as exc:
            logger.exception("Failed to reset document store")
            st.error(f"Impossible de réinitialiser l'index : {exc}")

    for doc in documents:
        col_left, col_right = st.columns([4, 1])
        with col_left:
            st.write(f"**{doc.original_name}**")
            chunk_count = len(doc.chunk_ids or [])
            stored_path = Path(doc.stored_path)
            size_label = "inconnu"
            if stored_path.exists():
                size_bytes = stored_path.stat().st_size
                size_label = f"{size_bytes/1024:.1f} Ko" if size_bytes < 1024 * 1024 else f"{size_bytes/1024/1024:.2f} Mo"
            st.caption(f"{chunk_count} chunks · {size_label} · .{doc.ext}")
            with st.expander("Aperçu"):
                if stored_path.exists():
                    try:
                        preview_text = stored_path.read_text(encoding="utf-8", errors="ignore").strip()
                        if len(preview_text) > DOC_PREVIEW_CHARS:
                            preview_text = preview_text[:DOC_PREVIEW_CHARS] + "..."
                        st.text(preview_text or "(fichier vide)")
                    except Exception as exc:
                        logger.exception("Failed to read preview", extra={"path": str(stored_path)})
                        st.error(f"Impossible de lire l'aperçu : {exc}")
                else:
                    st.warning("Fichier introuvable sur le disque.")
        with col_right:
            if st.button("Supprimer", key=f"delete_{doc.doc_id}"):
                try:
                    success = delete_document(doc.doc_id)
                    if not success:
                        st.error("Suppression impossible ou document introuvable.")
                    else:
                        st.rerun()
                except Exception as exc:
                    logger.exception("Failed to delete document", extra={"doc_id": doc.doc_id})
                    st.error(f"Impossible de supprimer ce document: {exc}")
