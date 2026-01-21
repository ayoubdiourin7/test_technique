from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.config import DEFAULT_TOP_K, LLM_MODEL_NAME, OPENAI_API_KEY
from rag.pipeline.contextualizer import contextualize_history, rewrite_question_with_history
from rag.pipeline.hybrid_retriever import HybridRetriever
from rag.pipeline.safety import sanitize_question

logger = logging.getLogger(__name__)

def _format_docs(docs: list[Any]) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        content = doc.page_content.strip()
        lines.append(f"[{idx}] {content}")
    return "\n\n".join(lines)


def _collect_sources(docs: list[Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata or {}
        sources.append(
            {
                "doc_id": meta.get("doc_id"),
                "source_path": meta.get("source_path"),
                "chunk_index": meta.get("chunk_index"),
                "doc_format": meta.get("doc_format"),
                "original_name": meta.get("original_name"),
            }
        )
    return sources


def _has_citation(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text or ""))


@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY is missing; cannot initialize ChatOpenAI.")
        raise ValueError("OPENAI_API_KEY is required to run the QA pipeline.")
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL_NAME,
        temperature=0,
    )


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    history: list[dict[str, Any]] | None = None,
    llm=None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Run retrieval-augmented QA and return (answer, sources).
    Sources are lightweight dicts with doc_id, source_path, chunk_index, doc_format.
    """
    cleaned_question = question.strip()
    if not cleaned_question:
        return "La requête est vide.", []

    history_records = history or []
    history_summary = contextualize_history(history_records)

    llm = llm or _get_llm()
    rewritten_question = rewrite_question_with_history(cleaned_question, history_records, llm)

    retriever = HybridRetriever(dense_k=top_k, lexical_k=top_k)
    docs = retriever.invoke(rewritten_question, k=top_k)
    if not docs:
        logger.warning("No documents available for retrieval.")
        return "Aucun document disponible pour répondre à la question.", []

    context = _format_docs(docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a legal assistant. Use the conversation summary and the retrieved context to answer. "
                "If the answer is not in the provided documents, say you cannot answer from the available documents. "
                "Keep responses concise and cite sources using [index] matching the context blocks.",
            ),
            (
                "user",
                "Conversation summary:\n{history_summary}\n\nQuestion: {question}\n\nContext:\n{context}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {"question": cleaned_question, "context": context, "history_summary": history_summary}
    )
    answer = answer.strip()

    if not _has_citation(answer):
        logger.warning("Answer missing citations; refusing to respond without sources.")
        return (
            "Je ne peux répondre que sur la base des documents disponibles et aucune citation n'a été fournie.",
            [],
        )

    sources = _collect_sources(docs)
    return answer, sources
