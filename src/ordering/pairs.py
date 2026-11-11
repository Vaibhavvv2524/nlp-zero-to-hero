"""Sentence-pair dataset for the mBERT ordering classifier.

The classifier answers one question: *given two sentences, does A come before
B?* Training pairs are built from consecutive sentences of a Wikipedia
article. Each consecutive pair produces a positive example ``(A, B, 1)`` and a
negative example ``(B, A, 0)``, so the label distribution is balanced.
"""

from ..constants import ORDER_MAX_LENGTH
from ..utils import read_jsonl


def make_pairs_from_sentences(sentences, max_pairs=30, prefix=""):
    """Turn ordered sentences into before/after training pairs.

    ``sentences`` is the list of ``{"words": [...], "labels": [...]}`` dicts
    produced by :mod:`src.data_prep`.
    """
    positions = list(range(max(0, len(sentences) - 1)))
    if max_pairs is not None and len(positions) > max_pairs:
        step = len(positions) / max_pairs
        positions = [positions[int(index * step)] for index in range(max_pairs)]

    examples = []
    for index in positions:
        first = sentences[index]
        second = sentences[index + 1]
        sentence_a = " ".join(first["words"]).strip()
        sentence_b = " ".join(second["words"]).strip()
        if not sentence_a or not sentence_b:
            continue
        pair_id = f"{prefix}-{index}"
        examples.append(
            {
                "id": pair_id,
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "label": 1,
            }
        )
        examples.append(
            {
                "id": pair_id,
                "sentence_a": sentence_b,
                "sentence_b": sentence_a,
                "label": 0,
            }
        )
    return examples


class SentencePairDataset:
    """Plain PyTorch-style dataset (no torch import needed)."""

    def __init__(self, path, tokenizer, max_length=ORDER_MAX_LENGTH):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoding = self.tokenizer(
            row["sentence_a"],
            row["sentence_b"],
            truncation=True,
            max_length=self.max_length,
        )
        item = {key: list(value) for key, value in encoding.items()}
        item["labels"] = int(row["label"])
        return item
