"""Shared configuration for the Hindi punctuation -> ordering -> translation pipeline.

Every path is absolute (derived from the repository root) so the same code
works from a script, a notebook or any working directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PUNCT_DATA = DATA_PROCESSED / "punctuation"
ORDER_DATA = DATA_PROCESSED / "ordering"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

PUNCT_MODEL_DIR = MODELS / "punctuation"
ORDER_MODEL_DIR = MODELS / "ordering"

# The only two model families used by the project (no LLM, no fallback model).
BASE_MODEL = "bert-base-multilingual-cased"
TRANSLATION_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"
SRC_LANG = "hin_Deva"
TGT_LANG = "eng_Latn"

# Punctuation labels, in the order expected by the classifier head.
LABELS = ["none", "comma", "sentence_end", "question", "exclamation"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
PUNCTUATION = {
    "none": "",
    "comma": ",",
    "sentence_end": "\u0964",  # Devanagari danda
    "question": "?",
    "exclamation": "!",
}
BOUNDARY_LABELS = {"sentence_end", "question", "exclamation"}
BOUNDARY_IDS = [LABEL2ID[label] for label in BOUNDARY_LABELS]

# Training data: Hindi Wikipedia (the assignment forbids LLM-generated data).
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.hi"

# Punctuation windows are measured in words, not subword tokens.
WINDOW_WORDS = 64
STRIDE_WORDS = 40
MIN_WORDS = 32
MAX_WORDS = 96
MIN_SENT_WORDS = 5
MAX_SENT_WORDS = 40
MAX_SEQ_LENGTH = 128

# Sentence-ordering pair classifier.
ORDER_MAX_LENGTH = 192

GROUP_SIZE = 5
MEMORY_LIMIT_GB = 1.5
SEED = 42
