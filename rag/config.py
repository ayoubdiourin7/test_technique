import os
import logging

# Disable Chroma telemetry BEFORE any other imports to prevent initialization issues
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configure basic logging once for the app.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHUNKS_DIR = DATA_DIR / "chunks"
CHROMA_DIR = DATA_DIR / "chroma"
REGISTRY_DB_PATH = DATA_DIR / "registry.sqlite3"
CONVERSATIONS_DB_PATH = DATA_DIR / "conversations.sqlite3"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDINGS_MODEL_NAME = os.getenv("OPENAI_EMBEDDINGS", "text-embedding-3-small")
LLM_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "4000"))
HYBRID_K = int(os.getenv("HYBRID_K", "8"))  # number of candidates to pull from each retriever
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "0.4"))  # weight for BM25 in fusion
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "12"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "1200"))
REWRITE_MAX_MESSAGES = int(os.getenv("REWRITE_MAX_MESSAGES", "6"))

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
USE_TIKTOKEN = os.getenv("USE_TIKTOKEN", "true").strip().lower() in {"1", "true", "yes", "y"}
DOC_PREVIEW_CHARS = int(os.getenv("DOC_PREVIEW_CHARS", "400"))

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
