"""Measure every stage of the assignment pipeline.

* punctuation: token accuracy, boundary precision/recall/F1, per-class F1;
* ordering: pair-classification accuracy plus Kendall's tau on shuffled
  held-out articles (with a TF-IDF-only ablation for comparison);
* translation: corpus BLEU and chrF on a small IndicTrans2 benchmark sample.

Usage
-----
    python -m src.evaluate
    python -m src.evaluate --skip-translation
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from .constants import (
    BOUNDARY_IDS,
    LABELS,
    LABEL2ID,
    MEMORY_LIMIT_GB,
    ORDER_DATA,
    ORDER_MODEL_DIR,
    PUNCT_DATA,
    PUNCT_MODEL_DIR,
    REPORTS,
    SEED,
    TRANSLATION_MODEL,
)
from .utils import ensure_dir, read_jsonl, save_json, set_seed


def _boundary_metrics(true_labels, predicted_labels):
    boundary_true = np.isin(true_labels, BOUNDARY_IDS)
    boundary_pred = np.isin(predicted_labels, BOUNDARY_IDS)
    precision, recall, f1, _ = precision_recall_fscore_support(
        boundary_true, boundary_pred, average="binary", zero_division=0
    )
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def evaluate_punctuation(
    model_dir=PUNCT_MODEL_DIR,
    test_path=PUNCT_DATA / "test.jsonl",
    max_examples=1500,
    memory_gb=MEMORY_LIMIT_GB,
):
    from .punctuation.predict import PunctuationRestorer

    rows = read_jsonl(test_path)
    if max_examples is not None:
        rows = rows[:max_examples]

    restorer = PunctuationRestorer(model_dir, memory_gb=memory_gb)
    true_labels = []
    predicted_labels = []
    for row in rows:
        predictions = restorer.predict_labels(row["words"])
        true_labels.extend(LABEL2ID[label] for label in row["labels"])
        predicted_labels.extend(predictions)
    restorer.close()

    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )
    return {
        "examples": len(rows),
        "words": int(len(true_labels)),
        "accuracy": round(
            float(accuracy_score(true_labels, predicted_labels)), 4
        ),
        "boundary": _boundary_metrics(true_labels, predicted_labels),
        "per_class_f1": {
            label: round(float(per_class_f1[index]), 4)
            for index, label in enumerate(LABELS)
        },
    }


def evaluate_ordering_pairs(
    scorer,
    test_path=ORDER_DATA / "test.jsonl",
    max_examples=3000,
):
    rows = read_jsonl(test_path)
    if max_examples is not None:
        rows = rows[:max_examples]
    pairs = [(row["sentence_a"], row["sentence_b"]) for row in rows]
    true_labels = np.asarray([row["label"] for row in rows])
    predicted_labels = np.asarray(scorer.predict_pair_labels(pairs))
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    return {
        "pairs": len(rows),
        "accuracy": round(
            float(accuracy_score(true_labels, predicted_labels)), 4
        ),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
    }


def evaluate_translation(
    translator,
    dataset_name="cfilt/iitb-english-hindi",
    max_examples=200,
):
    try:
        from datasets import load_dataset
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as error:
        return {"status": "skipped", "reason": str(error)}

    try:
        stream = load_dataset(dataset_name, split="test", streaming=True)
        sources = []
        references = []
        for row in stream:
            pair = row.get("translation") if isinstance(row, dict) else None
            if pair is None:
                pair = row
            source = pair.get("hi")
            reference = pair.get("en")
            if not source or not reference:
                continue
            sources.append(source)
            references.append(reference)
            if len(sources) >= max_examples:
                break
        if not sources:
            return {
                "status": "skipped",
                "reason": "dataset contained no hi/en sentence pairs",
            }

        hypotheses = translator.translate(sources)
        bleu = BLEU().corpus_score(hypotheses, [references]).score
        chrf = CHRF(word_order=2).corpus_score(hypotheses, [references]).score
        return {
            "status": "ok",
            "dataset": dataset_name,
            "examples": len(sources),
            "bleu": round(float(bleu), 3),
            "chrf": round(float(chrf), 3),
            "samples": [
                {
                    "hindi": sources[index],
                    "reference": references[index],
                    "hypothesis": hypotheses[index],
                }
                for index in range(min(5, len(sources)))
            ],
        }
    except Exception as error:  # noqa: BLE001 - evaluation is optional
        return {
            "status": "skipped",
            "reason": f"{type(error).__name__}: {error}",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--punctuation-model", type=Path, default=PUNCT_MODEL_DIR)
    parser.add_argument("--ordering-model", type=Path, default=ORDER_MODEL_DIR)
    parser.add_argument("--punct-test", type=Path, default=PUNCT_DATA / "test.jsonl")
    parser.add_argument("--order-test", type=Path, default=ORDER_DATA / "test.jsonl")
    parser.add_argument(
        "--order-eval", type=Path, default=ORDER_DATA / "ordering_eval.jsonl"
    )
    parser.add_argument("--max-punct-examples", type=int, default=1500)
    parser.add_argument("--max-order-pairs", type=int, default=3000)
    parser.add_argument("--max-order-articles", type=int, default=40)
    parser.add_argument("--max-translation-examples", type=int, default=200)
    parser.add_argument(
        "--translation-dataset", default="cfilt/iitb-english-hindi"
    )
    parser.add_argument("--translation-model", default=TRANSLATION_MODEL)
    parser.add_argument("--memory-gb", type=float, default=MEMORY_LIMIT_GB)
    parser.add_argument("--report", type=Path, default=REPORTS / "evaluation.json")
    parser.add_argument("--skip-punctuation", action="store_true")
    parser.add_argument("--skip-ordering", action="store_true")
    parser.add_argument("--skip-translation", action="store_true")
    args = parser.parse_args(argv)

    set_seed(SEED)
    report = {
        "no_llm": True,
        "memory_limit_gb": args.memory_gb,
        "punctuation": {"status": "skipped"},
        "ordering": {"status": "skipped"},
        "translation": {"status": "skipped"},
    }

    if not args.skip_punctuation:
        print("Evaluating punctuation ...")
        report["punctuation"] = {
            "status": "ok",
            **evaluate_punctuation(
                args.punctuation_model,
                args.punct_test,
                args.max_punct_examples,
                args.memory_gb,
            ),
        }
        print(report["punctuation"])

    if not args.skip_ordering:
        from .ordering.decode import OrderScorer, evaluate_orderings

        print("Evaluating ordering ...")
        scorer = OrderScorer(args.ordering_model, memory_gb=args.memory_gb)
        pair_metrics = evaluate_ordering_pairs(
            scorer, args.order_test, args.max_order_pairs
        )
        tau_metrics = evaluate_orderings(
            args.order_eval, scorer, args.max_order_articles
        )
        scorer.close()
        report["ordering"] = {
            "status": "ok",
            "pairs": pair_metrics,
            "kendall_tau": tau_metrics,
        }
        print(report["ordering"])

    if not args.skip_translation:
        from .translation.indictrans import IndicTrans2Translator

        print("Evaluating translation ...")
        translator = IndicTrans2Translator(
            args.translation_model, memory_gb=args.memory_gb
        )
        report["translation"] = evaluate_translation(
            translator,
            args.translation_dataset,
            args.max_translation_examples,
        )
        translator.close()
        print(report["translation"])

    ensure_dir(args.report.parent)
    save_json(args.report, report)
    print("Wrote", args.report)
    return report


if __name__ == "__main__":
    main()
