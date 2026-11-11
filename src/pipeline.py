"""End-to-end assignment pipeline.

For every ``data/raw/*.txt`` file:

1. restore punctuation and sentence boundaries with the trained mBERT + LoRA
   token classifier;
2. reorder the sentences with the trained mBERT pair classifier (Borda
   aggregation by default, TF-IDF clustering or beam search optional);
3. translate the ordered sentences to English with IndicTrans2;
4. write Hindi/English sentence files and groups of five.

Models are loaded one stage at a time and released before the next stage, so
the 1.5 GB GPU budget is respected.

Usage
-----
    python -m src.pipeline
    python -m src.pipeline --no-translate
    python -m src.pipeline --limit 1
"""

import argparse
import time
from pathlib import Path

from .constants import (
    DATA_RAW,
    GROUP_SIZE,
    MEMORY_LIMIT_GB,
    ORDER_MODEL_DIR,
    OUTPUTS,
    PUNCT_MODEL_DIR,
    REPORTS,
    SEED,
    TRANSLATION_MODEL,
)
from .utils import (
    ensure_dir,
    group_lines,
    save_json,
    set_seed,
    write_lines,
)


def read_tokens(path):
    """Read a space-separated assignment file and remove ``NULL`` tokens."""
    text = Path(path).read_text(encoding="utf-8")
    raw_tokens = text.split()
    words = [token for token in raw_tokens if token != "NULL"]
    return words, len(raw_tokens) - len(words)


def run(args):
    set_seed(SEED)
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)

    files = sorted(input_dir.glob("*.txt"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .txt files found in {input_dir.resolve()}")

    records = {}
    for path in files:
        words, null_count = read_tokens(path)
        records[path.stem] = {
            "words": words,
            "null_count": null_count,
        }
    print(f"Loaded {len(records)} files from {input_dir.resolve()}")

    # ------------------------------------------------------------------
    # Stage 1: punctuation + sentence boundaries
    # ------------------------------------------------------------------
    from .punctuation.predict import PunctuationRestorer

    print("\nStage 1/3 - punctuation restoration")
    restorer = PunctuationRestorer(
        args.punctuation_model, memory_gb=args.memory_gb
    )
    print("Punctuation memory:", restorer.memory())
    start = time.perf_counter()
    for stem, record in records.items():
        labels, sentences, plain_sentences = restorer.restore(record["words"])
        record["labels"] = labels
        record["sentences_hi"] = sentences
        record["sentences_plain"] = plain_sentences
        print(f"  {stem}: {len(record['words'])} words -> {len(sentences)} sentences")
    punctuation_seconds = time.perf_counter() - start
    restorer.close()
    del restorer

    # ------------------------------------------------------------------
    # Stage 2: sentence ordering
    # ------------------------------------------------------------------
    from .ordering.decode import order_sentences

    print("\nStage 2/3 - sentence ordering")
    from .ordering.decode import OrderScorer

    scorer = OrderScorer(args.ordering_model, memory_gb=args.memory_gb)
    print("Ordering memory:", scorer.memory())
    start = time.perf_counter()
    for stem, record in records.items():
        order = order_sentences(
            record["sentences_plain"],
            scorer,
            strategy=args.strategy,
            n_clusters=args.n_clusters,
            beam_width=args.beam_width,
        )
        record["ordered_hi"] = [record["sentences_hi"][index] for index in order]
        record["groups_hi"] = group_lines(record["ordered_hi"], GROUP_SIZE)
        print(f"  {stem}: ordered {len(order)} sentences")
    ordering_seconds = time.perf_counter() - start
    scorer.close()
    del scorer

    # ------------------------------------------------------------------
    # Stage 3: translation
    # ------------------------------------------------------------------
    translation_seconds = None
    if not args.no_translate:
        from .translation.indictrans import IndicTrans2Translator

        print("\nStage 3/3 - Hindi -> English translation")
        translator = IndicTrans2Translator(
            args.translation_model, memory_gb=args.memory_gb
        )
        print("Translation memory:", translator.memory())
        start = time.perf_counter()

        flat_sentences = []
        for record in records.values():
            flat_sentences.extend(record["ordered_hi"])
        translations = translator.translate(
            flat_sentences, batch_size=args.translation_batch_size
        )

        offset = 0
        for record in records.values():
            count = len(record["ordered_hi"])
            record["ordered_en"] = translations[offset : offset + count]
            record["groups_en"] = [
                " ".join(group)
                for group in group_lines(record["ordered_en"], GROUP_SIZE)
            ]
            offset += count

        translation_seconds = time.perf_counter() - start
        translator.close()
        del translator

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    summary = {
        "group_size": GROUP_SIZE,
        "ordering_strategy": args.strategy,
        "punctuation_model": str(args.punctuation_model),
        "ordering_model": str(args.ordering_model),
        "translation_model": None if args.no_translate else args.translation_model,
        "memory_limit_gb": args.memory_gb,
        "timings_seconds": {
            "punctuation": round(punctuation_seconds, 2),
            "ordering": round(ordering_seconds, 2),
            "translation": (
                None if translation_seconds is None else round(translation_seconds, 2)
            ),
        },
        "files": {},
    }

    for stem, record in records.items():
        write_lines(output_dir / f"{stem}.sentences_hi.txt", record["ordered_hi"])
        write_lines(
            output_dir / f"{stem}.groups_hi.txt",
            [" ".join(group) for group in record["groups_hi"]],
        )
        file_summary = {
            "words": len(record["words"]),
            "null_tokens_removed": record["null_count"],
            "sentences": len(record["ordered_hi"]),
            "groups": len(record["groups_hi"]),
        }
        if not args.no_translate:
            write_lines(
                output_dir / f"{stem}.sentences_en.txt", record["ordered_en"]
            )
            write_lines(
                output_dir / f"{stem}.groups_en.txt", record["groups_en"]
            )
        summary["files"][stem] = file_summary
        print(f"  wrote {stem}: {file_summary}")

    save_json(REPORTS / "run_summary.json", summary)
    print("\nOutputs:", output_dir.resolve())
    print("Summary:", (REPORTS / "run_summary.json").resolve())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--punctuation-model", type=Path, default=PUNCT_MODEL_DIR)
    parser.add_argument("--ordering-model", type=Path, default=ORDER_MODEL_DIR)
    parser.add_argument("--translation-model", default=TRANSLATION_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--strategy",
        choices=("borda", "clusters", "beam"),
        default="borda",
        help="Sentence decoder: pairwise Borda score (default), TF-IDF clusters or beam chain.",
    )
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--translation-batch-size", type=int, default=4)
    parser.add_argument("--memory-gb", type=float, default=MEMORY_LIMIT_GB)
    parser.add_argument("--no-translate", action="store_true")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
