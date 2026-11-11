"""End-to-end baseline: segment -> order -> group of 5 -> optional translation.

Run stages 1-2 locally (no torch needed):

    python -m src.pipeline

Run with translation on Colab (after installing transformers + sentencepiece):

    python -m src.pipeline --translate
"""

import argparse
import json
from pathlib import Path

from .hindi_utils import tokenize_file
from .order import evaluate_synthetic, order_sentences
from .segment import punctuate, segment_tokens

GROUP_SIZE = 5


def process_file(path, add_commas=True, cluster=True):
    tokens, null_positions = tokenize_file(path)
    raw_sentences = segment_tokens(tokens, null_positions)
    plain = [" ".join(sentence) for sentence in raw_sentences]
    indices = (
        order_sentences(plain) if cluster else list(range(len(plain)))
    )
    ordered_plain = [plain[index] for index in indices]
    ordered_punct = [
        punctuate(raw_sentences[index], add_commas=add_commas) for index in indices
    ]
    groups = [
        ordered_punct[start : start + GROUP_SIZE]
        for start in range(0, len(ordered_punct), GROUP_SIZE)
    ]
    return {
        "path": path,
        "words": len(tokens),
        "nulls": len(null_positions),
        "sentences": len(raw_sentences),
        "ordered_plain": ordered_plain,
        "ordered_punct": ordered_punct,
        "groups": groups,
    }


def write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="baseline/baseline_outputs")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-hi-en")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-commas", action="store_true")
    parser.add_argument("--no-cluster", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .txt files found in {input_dir.resolve()}")

    translator = None
    if args.translate:
        from .translate import OpusTranslator

        translator = OpusTranslator(args.model, args.device)
        print("Loaded translator:", args.model, "on", translator.device)
        print("Memory:", translator.memory())

    summary = []
    for path in files:
        result = process_file(
            path, add_commas=not args.no_commas, cluster=not args.no_cluster
        )
        stem = path.stem
        write_lines(output_dir / f"{stem}.sentences.txt", result["ordered_punct"])
        write_lines(output_dir / f"{stem}.groups_hi.txt", [
            " ".join(group) for group in result["groups"]
        ])

        if translator is not None:
            per_sentence = translator.translate(result["ordered_punct"])
            write_lines(output_dir / f"{stem}.translation_sentences.txt", per_sentence)
            per_group = translator.translate(
                [" ".join(group) for group in result["groups"]]
            )
            write_lines(output_dir / f"{stem}.translation_groups.txt", per_group)
            print("Memory after", stem, ":", translator.memory())

        summary.append(
            {
                "file": path.name,
                "words": result["words"],
                "nulls": result["nulls"],
                "sentences": result["sentences"],
                "groups": len(result["groups"]),
            }
        )
        print(
            f"{path.name}: words={result['words']} sentences={result['sentences']} "
            f"groups={len(result['groups'])}"
        )

    tau = evaluate_synthetic()
    report = {
        "group_size": GROUP_SIZE,
        "synthetic_order_kendall_tau": round(tau, 4),
        "files": summary,
    }
    reports_dir = Path("baseline")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "baseline_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if translator is not None:
        translator.close()

    print("Synthetic ordering Kendall tau:", round(tau, 4))
    print("Wrote outputs to", output_dir.resolve())


if __name__ == "__main__":
    main()
