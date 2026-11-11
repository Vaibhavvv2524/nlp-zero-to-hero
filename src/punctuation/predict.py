"""Inference for the punctuation model.

Long token streams are processed with overlapping windows. Predictions for
every word are averaged with a triangular weight, so words near a window edge
(which have less context) count less than words in the middle.
"""

from pathlib import Path

import numpy as np

from ..constants import (
    BASE_MODEL,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    MAX_SEQ_LENGTH,
    MEMORY_LIMIT_GB,
    PUNCT_MODEL_DIR,
    STRIDE_WORDS,
    WINDOW_WORDS,
)
from ..utils import (
    free_memory,
    get_device,
    gpu_memory,
    limit_gpu_memory,
    restore_punctuation,
    sliding_windows,
    split_into_sentences,
    window_weight,
)

INFERENCE_MAX_LENGTH = max(MAX_SEQ_LENGTH, 160)


class PunctuationRestorer:
    """Load the LoRA adapter and predict one punctuation label per word."""

    def __init__(
        self,
        model_dir=PUNCT_MODEL_DIR,
        device=None,
        memory_gb=MEMORY_LIMIT_GB,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.memory_report = limit_gpu_memory(memory_gb)
        self.device = device or get_device()

        model_dir = Path(model_dir)
        adapter_dir = model_dir / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"No punctuation adapter found at {adapter_dir}. "
                "Train it first: python -m src.punctuation.train"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base_model = AutoModelForTokenClassification.from_pretrained(
            BASE_MODEL,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()

    def predict_labels(self, words, batch_size=16):
        """Return one label id per input word."""
        import torch

        words = list(words)
        if not words:
            return []

        windows = sliding_windows(len(words), WINDOW_WORDS, STRIDE_WORDS)
        texts = [words[start:end] for start, end in windows]

        scores = np.zeros((len(words), len(LABELS)), dtype=np.float64)
        weights = np.zeros(len(words), dtype=np.float64)

        with torch.inference_mode():
            for batch_start in range(0, len(texts), batch_size):
                batch_texts = texts[batch_start : batch_start + batch_size]
                batch_windows = windows[batch_start : batch_start + batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=INFERENCE_MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                )
                word_ids_per_row = [
                    encoding.word_ids(row) for row in range(len(batch_texts))
                ]
                encoding = {
                    key: value.to(self.device)
                    for key, value in encoding.items()
                }
                logits = self.model(**encoding).logits.float()
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

                for row, (start, end) in enumerate(batch_windows):
                    previous_word = None
                    for position, word_id in enumerate(word_ids_per_row[row]):
                        if word_id is None or word_id == previous_word:
                            continue
                        previous_word = word_id
                        weight = window_weight(word_id, end - start)
                        scores[start + word_id] += (
                            weight * probabilities[row, position]
                        )
                        weights[start + word_id] += weight

        labels = []
        for index in range(len(words)):
            if weights[index] == 0:
                labels.append(LABEL2ID["none"])
            else:
                labels.append(int(np.argmax(scores[index] / weights[index])))
        return labels

    def restore(self, words):
        """Return ``(label_ids, punctuated_sentences, plain_sentences)``."""
        label_ids = self.predict_labels(words)
        label_names = [ID2LABEL[label_id] for label_id in label_ids]
        punctuated = split_into_sentences(words, label_names, with_punctuation=True)
        plain = split_into_sentences(words, label_names, with_punctuation=False)
        return label_ids, punctuated, plain

    def restore_text(self, words):
        label_ids = self.predict_labels(words)
        label_names = [ID2LABEL[label_id] for label_id in label_ids]
        return restore_punctuation(words, label_names)

    def close(self):
        del self.model
        self.model = None
        free_memory()

    def memory(self):
        return gpu_memory()
