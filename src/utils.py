"""Small helpers shared by every stage.

Heavy libraries (torch, transformers, datasets) are imported inside the
functions that need them, so data preparation, evaluation maths and the CLI
help screens work without a GPU stack installed.
"""

import gc
import json
import random
import re
from pathlib import Path

import numpy as np

from .constants import (
    BOUNDARY_LABELS,
    MEMORY_LIMIT_GB,
    PUNCTUATION,
    SEED,
)

_TRAILING_QUOTES = "\"'”’)]}»"
_LEADING_QUOTES = "\"'“‘([{«"


# ---------------------------------------------------------------------------
# Reproducibility and devices
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    """Seed Python, NumPy and (when available) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return ``cuda``, ``mps`` or ``cpu`` for the current machine."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def attention_implementation():
    """Pick a safe BERT attention backend for the current device.

    PyTorch's MPS backend does not implement dropout inside
    ``scaled_dot_product_attention`` (transformers 4.x uses SDPA by default),
    so training on Apple Silicon must fall back to eager attention. CUDA keeps
    the faster default.
    """
    import torch

    if torch.cuda.is_available():
        return None
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "eager"
    return None


def limit_gpu_memory(gb=MEMORY_LIMIT_GB):
    """Hard-cap this process at ``gb`` gigabytes of GPU memory.

    Returns a small report. On CPU/MPS this is a no-op because the assignment
    memory budget is enforced on Colab's CUDA GPU only.
    """
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu/mps", "limit_gb": None, "total_gb": None}

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gb is not None and gb > 0:
        fraction = min(1.0, float(gb) / total_gb)
        torch.cuda.set_per_process_memory_fraction(fraction)
        return {
            "device": torch.cuda.get_device_name(0),
            "limit_gb": round(float(gb), 3),
            "total_gb": round(total_gb, 3),
            "fraction": round(fraction, 4),
        }
    return {
        "device": torch.cuda.get_device_name(0),
        "limit_gb": None,
        "total_gb": round(total_gb, 3),
    }


def gpu_memory():
    """Return current / reserved / peak GPU memory in gigabytes."""
    import torch

    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 4),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 4),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
            "total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 3
            ),
        }

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return {
            "device": "mps",
            "allocated_gb": round(torch.mps.current_allocated_memory() / 1024**3, 4),
        }
    return {"device": "cpu"}


def free_memory():
    """Delete cached tensors so the next stage can reuse the memory budget."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def save_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def write_lines(path, lines):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Hindi / Devanagari helpers
# ---------------------------------------------------------------------------
def has_devanagari(text):
    return any("\u0900" <= char <= "\u097f" for char in text)


def _strip_edge_quotes(token):
    word = token.strip()
    changed = True
    while changed and word:
        changed = False
        if word and word[-1] in _TRAILING_QUOTES:
            word = word[:-1]
            changed = True
        if word and word[0] in _LEADING_QUOTES:
            word = word[1:]
            changed = True
    return word


def strip_punctuation(token):
    """Split one raw token into ``(clean_word, punctuation_label)``.

    The function understands the marks that appear in Hindi Wikipedia text and
    in the assignment files. It never uses a ``\\w`` regex, which would break
    Devanagari matras.
    """
    word = _strip_edge_quotes(token)
    if not word:
        return "", "none"

    last = word[-1]
    if last in "।॥":
        while word and word[-1] in "।॥":
            word = word[:-1]
        return word, "sentence_end"
    if last == ",":
        while word and word[-1] == ",":
            word = word[:-1]
        return word, "comma"
    if last == "?":
        return word[:-1], "question"
    if last == "!":
        return word[:-1], "exclamation"
    if last == "." and not re.fullmatch(r"\d+(\.\d+)*", word):
        return word[:-1], "sentence_end"
    return word, "none"


def restore_punctuation(words, labels):
    """Attach the predicted punctuation mark to every word."""
    pieces = []
    for word, label in zip(words, labels):
        pieces.append(word + PUNCTUATION[label])
    return " ".join(pieces)


def split_into_sentences(words, labels, with_punctuation=True):
    """Rebuild sentence strings using the predicted boundary labels.

    Set ``with_punctuation=False`` to get plain word sequences, which is the
    format the ordering model was trained on.
    """
    sentences = []
    current = []
    for word, label in zip(words, labels):
        if with_punctuation:
            current.append(word + PUNCTUATION[label])
        else:
            current.append(word)
        if label in BOUNDARY_LABELS:
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return sentences


def sliding_windows(length, size, stride):
    """Return ``(start, end)`` windows covering ``length`` items."""
    if length <= 0:
        return []
    if length <= size:
        return [(0, length)]
    windows = []
    start = 0
    while start < length:
        end = min(start + size, length)
        windows.append((start, end))
        if end == length:
            break
        start += stride
    return windows


def window_weight(position, length):
    """Triangular weight: predictions near a window edge count less."""
    if length <= 1:
        return 1.0
    center = (length - 1) / 2.0
    distance = abs(position - center) / (center + 1.0)
    return max(0.05, 1.0 - distance)


def group_lines(lines, size):
    """Group a flat list into consecutive chunks of ``size``."""
    return [lines[start : start + size] for start in range(0, len(lines), size)]
