"""Unicode-safe helpers for the Devanagari assignment files.

The input files are space-separated tokens. ``str.split`` is used instead of a
``\\w`` regex because Python's ``\\w`` does not include Devanagari combining
marks (matras), so ``re.findall(r"\\w+", "अयोध्या")`` returns broken pieces.
"""

from pathlib import Path

NULL_TAG = "NULL"


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def tokenize(text):
    """Return (tokens, null_positions).

    ``NULL`` tokens are removed from the word stream but their gap positions are
    kept so the segmenter can treat them as weak boundary hints.
    """
    tokens = []
    null_positions = []
    for token in text.split():
        if token == NULL_TAG:
            null_positions.append(len(tokens))
        else:
            tokens.append(token)
    return tokens, null_positions


def tokenize_file(path):
    return tokenize(read_text(path))


def has_devanagari(text):
    return any("\u0900" <= char <= "\u097f" for char in text)
