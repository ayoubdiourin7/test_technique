from __future__ import annotations

import csv
import logging
from bs4 import BeautifulSoup
from pathlib import Path

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    # Simple normalization to keep output compact and readable.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _preprocess_txt(path: Path) -> str:
    return _clean_text(path.read_text(encoding="utf-8", errors="ignore"))


def _preprocess_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open(encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader, [])
        headers = [header.strip() for header in headers]
        for row in reader:
            pairs = []
            for idx, cell in enumerate(row):
                value = cell.strip()
                if not value:
                    continue
                label = headers[idx] if idx < len(headers) and headers[idx] else f"col_{idx + 1}"
                pairs.append(f"{label}: {value}")
            row_text = " | ".join(pairs)
            if row_text:
                rows.append(row_text)
    return _clean_text("\n".join(rows))


def _preprocess_html_for_rag(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    # Remove non-content noise
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # remove common cookie / modal / popup blocks if present
    noise_selectors = [
        "[id*='cookie']",
        "[class*='cookie']",
        "[id*='consent']",
        "[class*='consent']",
        "[id*='modal']",
        "[class*='modal']",
        "[id*='popup']",
        "[class*='popup']",
        "[class*='newsletter']",
    ]
    for sel in noise_selectors:
        for el in soup.select(sel):
            el.decompose()

    text = soup.get_text(separator="\n")
    return _clean_text(text)


def _preprocess_html(path: Path) -> str:
    return _preprocess_html_for_rag(path)


def preprocess_file(path: str | Path) -> str:
    """
    Load a file and return a simple text-only representation suitable for embeddings.
    Supported extensions: .txt, .csv, .html, .htm
    """
    file_path = Path(path)
    ext = file_path.suffix.lower().lstrip(".")

    if ext == "txt":
        return _preprocess_txt(file_path)
    if ext == "csv":
        return _preprocess_csv(file_path)
    if ext in {"html", "htm"}:
        return _preprocess_html(file_path)

    logger.error("Unsupported file extension encountered", extra={"path": str(file_path)})
    raise ValueError(f"Unsupported file extension: {file_path.suffix}")
