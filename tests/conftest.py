import sys
from pathlib import Path

# Ensure the project root (containing the rag package) is on the import path.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
