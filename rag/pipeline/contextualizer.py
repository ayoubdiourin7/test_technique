from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from rag.config import HISTORY_MAX_CHARS, HISTORY_MAX_MESSAGES, REWRITE_MAX_MESSAGES


def contextualize_history(
    history: list[dict[str, Any]],
    *,
    max_messages: int = HISTORY_MAX_MESSAGES,
    max_chars: int = HISTORY_MAX_CHARS,
) -> str:
    """
    Deterministically compress chat history into a short string for context.
    Skips empty lines and keeps the most recent exchanges, represented as Human/AI messages.
    """
    if not history:
        return "Aucun historique pertinent."

    safe_lines: list[str] = []
    for msg in history[-max_messages:]:
        role = msg.get("role", "").strip().lower()
        content = (msg.get("content", "") or "").strip()
        if not content:
            continue
        role_label = "User" if role == "user" else "Assistant" if role == "assistant" else "Message"
        safe_lines.append(f"{role_label}: {content}")

    if not safe_lines:
        return "Aucun historique pertinent."

    summary = " | ".join(safe_lines)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + " ..."
    return summary


def rewrite_question_with_history(
    question: str,
    history: list[dict[str, Any]],
    llm,
    *,
    max_messages: int = REWRITE_MAX_MESSAGES,
):
    """
    Ask the LLM to rewrite the latest user question to resolve pronouns and references.
    Falls back to the original question on any failure or missing history.
    """
    if not history:
        return question

    normalized: list[HumanMessage | AIMessage] = []
    for msg in history[-max_messages:]:
        role = msg.get("role", "").strip().lower()
        content = (msg.get("content", "") or "").strip()
        if not content:
            continue
        if role == "assistant":
            normalized.append(AIMessage(content=content))
        else:
            normalized.append(HumanMessage(content=content))

    if not normalized:
        return question

    system_msg = SystemMessage(
        content=(
            "Rewrite the user's latest question so that all pronouns and ambiguous references "
            "are replaced with explicit entities from the conversation. "
            "Return only the rewritten question. Do not answer the question."
        )
    )
    convo_lines = [f"{m.type.capitalize()}: {m.content}" for m in normalized]
    user_msg = HumanMessage(
        content=(
            "Conversation (oldest to newest):\n"
            f"{chr(10).join(convo_lines)}\n\n"
            f"Latest user question: {question}"
        )
    )

    try:
        resp = llm.invoke([system_msg, *normalized, user_msg])
        rewritten_raw = resp.content.strip()
        if not rewritten_raw:
            return question
        return " ".join(rewritten_raw.split())  # normalize whitespace
    except Exception:
        logging.exception("Failed to rewrite question; falling back to original.")
        return question
