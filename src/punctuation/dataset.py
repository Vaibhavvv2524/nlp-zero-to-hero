"""Token-classification dataset for punctuation restoration."""

from ..constants import LABEL2ID, MAX_SEQ_LENGTH
from ..utils import read_jsonl


class PunctuationDataset:
    """Wrap punctuation JSONL examples with word-level BERT subword labels.

    BERT splits a Hindi word into several subword tokens. Only the first
    subword carries the word's punctuation label; every other subword is set to
    ``-100`` so the loss ignores it. This keeps every word supervised exactly
    once.
    """

    def __init__(self, path, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoding = self.tokenizer(
            row["words"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoding.word_ids()
        labels = [-100] * len(encoding["input_ids"])
        previous_word = None
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word:
                continue
            previous_word = word_id
            labels[position] = LABEL2ID[row["labels"][word_id]]

        item = {key: list(value) for key, value in encoding.items()}
        item["labels"] = labels
        return item
