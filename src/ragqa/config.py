"""Paths and fixed settings shared by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

# SQuAD 2.0 development set (CC BY-SA 4.0). The official test set is hidden, so the dev set
# is the public labelled split. Nothing in this project is trained on it.
DATA_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
DATA_FILE = DATA_DIR / "dev-v2.0.json"
DATA_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"

# Chunking: word windows inside a paragraph, never across paragraphs.
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30

TOP_KS = (1, 3, 5, 10, 20)
