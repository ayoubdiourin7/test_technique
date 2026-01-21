from __future__ import annotations

import re
from typing import Iterable, Tuple

from rag.config import MAX_INPUT_LENGTH

INJECTION_PATTERNS: tuple[str, ...] = (
    # English
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard above instructions",
    "forget previous instructions",
    "forget all previous instructions",
    "reset your instructions",
    "you are now",
    "act as an unfiltered",
    "act as system",
    "system prompt",
    "developer mode",
    "jailbreak",
    "prompt injection",
    "roleplay as",
    "pretend to be",
    # French
    "ignore les instructions",
    "ignore toutes les instructions",
    "oublie les instructions",
    "oublie tout ce qui precede",
    "tu es maintenant",
    "tu es desormais",
    "agis comme",
    "role systeme",
    "mode developpeur",
    "sans filtre",
    "contourner la politique",
    "prompt injection",
)

NON_PRINTABLE_RE = re.compile(r"[^\x20-\x7E\n\r\t]")


def _contains_injection(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def sanitize_question(raw: str, *, raise_on_refusal: bool = True) -> str | Tuple[str | None, str | None]:
    """
    Normalize user input and block obvious prompt injection attempts.

    - Trim leading/trailing spaces.
    - Remove non-printable/invisible characters.
    - Enforce a maximum length.
    - Detect common override attempts in EN/FR.

    Raises ValueError on empty or unsafe input.
    """
    def _fail(message: str):
        if raise_on_refusal:
            raise ValueError(message)
        return None, message

    if raw is None:
        return _fail("La requête est vide.")

    cleaned = NON_PRINTABLE_RE.sub("", raw).strip()
    cleaned = " ".join(cleaned.split())  # normalize internal whitespace

    if not cleaned:
        return _fail("La requête est vide.")
    if len(cleaned) > MAX_INPUT_LENGTH:
        return _fail("La requête est trop longue.")
    if _contains_injection(cleaned, INJECTION_PATTERNS):
        return _fail("La requête a été refusée pour raisons de sécurité.")

    return cleaned if raise_on_refusal else (cleaned, None)
