"""Build the training and evaluation datasets from Hindi Wikipedia.

Two datasets are produced from the same articles (so the train/validation/test
split is identical for both stages):

``data/processed/punctuation/{train,val,test}.jsonl``
    Windows of 32-96 words. Every word has one of five punctuation labels.
    Sentences inside a window are shuffled, which mimics the assignment input
    (sentences arrive in random order) and stops the model from relying on
    cross-sentence continuity.

``data/processed/ordering/{train,val,test}.jsonl``
    Consecutive ``(sentence_a, sentence_b)`` pairs labelled 1 when ``a`` really
    comes first, plus the swapped pair labelled 0.

``data/processed/ordering/ordering_eval.jsonl``
    Complete held-out articles in their original order. Used to measure
    Kendall's tau after shuffling and re-ordering.

Usage
-----
    python -m src.data_prep --num-articles 3000

The Wikipedia download is streamed, so only the requested articles are stored.
"""

import argparse
import random
import re
from collections import Counter

from .constants import (
    BOUNDARY_LABELS,
    DATA_PROCESSED,
    MAX_SENT_WORDS,
    MAX_WORDS,
    MIN_SENT_WORDS,
    MIN_WORDS,
    ORDER_DATA,
    PUNCT_DATA,
    REPORTS,
    SEED,
    STRIDE_WORDS,
    WIKI_CONFIG,
    WIKI_DATASET,
    WINDOW_WORDS,
)
from .ordering.pairs import make_pairs_from_sentences
from .utils import (
    ensure_dir,
    has_devanagari,
    save_json,
    strip_punctuation,
    write_jsonl,
)

ARTICLE_PATTERN = re.compile(r"(?<=[।॥?!])\s+")


def parse_article(text, article_id):
    """Split raw Wikipedia text into clean, punctuation-labelled sentences."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    sentences = []
    for part in ARTICLE_PATTERN.split(text):
        words = []
        labels = []
        for token in part.split():
            word, label = strip_punctuation(token)
            if not word:
                continue
            if not has_devanagari(word) and not any(char.isdigit() for char in word):
                continue
            words.append(word)
            labels.append(label)

        current_words = []
        current_labels = []
        for word, label in zip(words, labels):
            current_words.append(word)
            current_labels.append(label)
            if label in BOUNDARY_LABELS:
                if MIN_SENT_WORDS <= len(current_words) <= MAX_SENT_WORDS:
                    sentences.append(
                        {"words": current_words, "labels": current_labels}
                    )
                current_words = []
                current_labels = []

    if len(sentences) < 3:
        return None
    return {"id": article_id, "sentences": sentences}


def load_wikipedia(num_articles, scan_multiplier=3):
    """Stream Hindi Wikipedia and keep the first ``num_articles`` valid pages."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "The 'datasets' package is required. Install requirements-colab.txt."
        ) from error

    try:
        stream = load_dataset(
            WIKI_DATASET, WIKI_CONFIG, split="train", streaming=True
        )
    except Exception as error:  # noqa: BLE001 - show the upstream message
        raise RuntimeError(
            f"Could not load {WIKI_DATASET} [{WIKI_CONFIG}]. "
            "Check the dataset id and your internet connection."
        ) from error

    articles = []
    scanned = 0
    limit = max(num_articles * scan_multiplier, num_articles)
    for row in stream:
        scanned += 1
        if scanned > limit or len(articles) >= num_articles:
            break
        article = parse_article(row.get("text") or "", f"hi-{len(articles):05d}")
        if article is not None:
            articles.append(article)

    if not articles:
        raise RuntimeError("No usable articles were found in the Wikipedia dump.")
    print(f"Kept {len(articles)} articles out of {scanned} scanned pages.")
    return articles


def make_punctuation_windows(article, split, rng):
    """Slide a word window over an article and shuffle the sentences inside."""
    sentences = article["sentences"]
    examples = []
    index = 0
    while index < len(sentences):
        end = index
        total = 0
        chunk = []
        while end < len(sentences):
            length = len(sentences[end]["words"])
            if chunk and total + length > MAX_WORDS:
                break
            chunk.append(sentences[end])
            total += length
            end += 1
            if total >= WINDOW_WORDS:
                break

        if MIN_WORDS <= total <= MAX_WORDS and len(chunk) >= 2:
            order = list(range(len(chunk)))
            rng.shuffle(order)
            words = []
            labels = []
            for position in order:
                words.extend(chunk[position]["words"])
                labels.extend(chunk[position]["labels"])
            examples.append(
                {
                    "id": f"{article['id']}-{index}",
                    "article_id": article["id"],
                    "split": split,
                    "words": words,
                    "labels": labels,
                }
            )

        advance = 0
        next_index = index
        while next_index < end and advance < STRIDE_WORDS:
            advance += len(sentences[next_index]["words"])
            next_index += 1
        index = max(index + 1, next_index)
    return examples


def split_article_ids(articles, seed=SEED, val_ratio=0.05, test_ratio=0.05):
    ids = [article["id"] for article in articles]
    rng = random.Random(seed)
    rng.shuffle(ids)
    count = len(ids)
    n_test = max(1, int(count * test_ratio))
    n_val = max(1, int(count * val_ratio))
    if count - n_test - n_val < 1:
        raise RuntimeError("Not enough articles to create train/val/test splits.")
    return {
        "train": set(ids[: count - n_test - n_val]),
        "val": set(ids[count - n_test - n_val : count - n_test]),
        "test": set(ids[count - n_test :]),
    }


def subset(articles, ids):
    return [article for article in articles if article["id"] in ids]


def cap_examples(rows, limit, seed=SEED):
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


def label_distribution(examples):
    counter = Counter()
    for example in examples:
        counter.update(example["labels"])
    total = sum(counter.values())
    return {
        label: {
            "count": counter.get(label, 0),
            "share": round(counter.get(label, 0) / max(total, 1), 5),
        }
        for label in ("none", "comma", "sentence_end", "question", "exclamation")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-articles", type=int, default=3000)
    parser.add_argument("--max-pairs-per-article", type=int, default=30)
    parser.add_argument("--max-train-pairs", type=int, default=40000)
    parser.add_argument("--max-train-windows", type=int, default=25000)
    parser.add_argument("--max-eval-pairs", type=int, default=3000)
    parser.add_argument("--eval-articles", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-ordering", action="store_true")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    print(f"Streaming {WIKI_DATASET} [{WIKI_CONFIG}] ...")
    articles = load_wikipedia(args.num_articles)
    splits = split_article_ids(articles, seed=args.seed)

    ensure_dir(PUNCT_DATA)
    ensure_dir(ORDER_DATA)

    punctuation_stats = {}
    ordering_stats = {}
    eval_articles = []

    for split in ("train", "val", "test"):
        split_articles = subset(articles, splits[split])
        rng = random.Random(args.seed + {"train": 0, "val": 1, "test": 2}[split])

        punct_examples = []
        window_limit = (
            args.max_train_windows if split == "train" else None
        )
        for article in split_articles:
            punct_examples.extend(make_punctuation_windows(article, split, rng))
            if window_limit and len(punct_examples) > window_limit * 1.2:
                break
        punct_examples = cap_examples(punct_examples, window_limit, args.seed)

        write_jsonl(PUNCT_DATA / f"{split}.jsonl", punct_examples)
        punctuation_stats[split] = {
            "articles": len(split_articles),
            "examples": len(punct_examples),
            "labels": label_distribution(punct_examples),
        }
        print(
            f"punctuation/{split}: {len(punct_examples)} windows "
            f"from {len(split_articles)} articles"
        )

        if args.skip_ordering:
            continue

        pair_examples = []
        for article in split_articles:
            pair_examples.extend(
                make_pairs_from_sentences(
                    article["sentences"],
                    max_pairs=args.max_pairs_per_article,
                    prefix=article["id"],
                )
            )
        limit = (
            args.max_train_pairs
            if split == "train"
            else args.max_eval_pairs
        )
        pair_examples = cap_examples(pair_examples, limit, args.seed)
        write_jsonl(ORDER_DATA / f"{split}.jsonl", pair_examples)
        ordering_stats[split] = {
            "articles": len(split_articles),
            "pairs": len(pair_examples),
        }
        print(
            f"ordering/{split}: {len(pair_examples)} pairs "
            f"from {len(split_articles)} articles"
        )

        if split == "test":
            for article in split_articles:
                count = len(article["sentences"])
                if 8 <= count <= 70:
                    eval_articles.append(
                        {
                            "article_id": article["id"],
                            "sentences": [
                                " ".join(sentence["words"])
                                for sentence in article["sentences"]
                            ],
                        }
                    )
                if len(eval_articles) >= args.eval_articles:
                    break

    if not args.skip_ordering:
        write_jsonl(ORDER_DATA / "ordering_eval.jsonl", eval_articles)
        print(f"ordering_eval: {len(eval_articles)} held-out articles")

    summary = {
        "dataset": WIKI_DATASET,
        "config": WIKI_CONFIG,
        "seed": args.seed,
        "articles": len(articles),
        "splits": {name: len(ids) for name, ids in splits.items()},
        "punctuation": punctuation_stats,
        "ordering": ordering_stats,
    }
    save_json(DATA_PROCESSED / "data_prep_summary.json", summary)
    save_json(REPORTS / "data_prep_summary.json", summary)
    print("Wrote", DATA_PROCESSED)
    return summary


if __name__ == "__main__":
    main()
