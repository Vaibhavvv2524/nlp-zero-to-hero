"""Fail loudly when a full Colab run is incomplete or clearly wrong.

This is the machine-checkable version of the test plan in Appendix I. It
verifies the data counts, adapter files, training/evaluation reports and the
end-to-end output alignment. It exits with status 1 and prints every failure
instead of silently accepting a half-finished run.

Usage
-----
    python scripts/verify_colab_run.py
    python scripts/verify_colab_run.py --skip-model-smoke
    python scripts/verify_colab_run.py --skip-models --skip-reports --skip-outputs
    python scripts/verify_colab_run.py --allow-skipped-translation

The notebook uses this at three points: after data preparation, after
evaluation (to catch bad metrics before running the pipeline) and at the end.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    LABELS,
    MAX_WORDS,
    MIN_WORDS,
    SEED,
)
from src.utils import read_jsonl, set_seed  # noqa: E402

EXPECTED_STEMS = [f"sample-{index}" for index in range(1, 8)]
SPLITS = ("train", "val", "test")
EXPECTED_SPLITS = {"train": 2700, "val": 150, "test": 150}
EXPECTED_ARTICLES = 3000
EXPECTED_PUNCT_TRAIN_WINDOWS = 25000
EXPECTED_ORDER_TRAIN_PAIRS = 40000
EXPECTED_ORDER_EVAL_ARTICLES = 40

MIN_PUNCT_ACCURACY = 0.85
MIN_BOUNDARY_F1 = 0.85
MIN_NO_PUNCT_F1 = 0.85
MIN_SENTENCE_END_F1 = 0.80
MIN_ORDER_ACCURACY = 0.58
MIN_ORDER_MACRO_F1 = 0.55
MIN_BORDA_TAU = 0.08
MIN_TRAIN_BOUNDARY_F1 = 0.75
MIN_TRAIN_ORDER_ACCURACY = 0.58
MIN_BLEU = 30.0
MIN_CHRF = 50.0


class Checker:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, condition, message, detail=None):
        self.checks += 1
        if condition:
            print(f"  ok   {message}")
            return True
        self.failures.append(message)
        suffix = "" if detail is None else f" [{detail}]"
        print(f"  FAIL {message}{suffix}")
        return False

    def section(self, title):
        print(f"\n== {title} ==")


def _load_json(path, checker, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - report every missing artifact
        checker.check(False, f"{label} is readable", f"{path}: {error}")
        return None


def check_data(root, checker):
    checker.section("data")
    processed = Path(root) / "data" / "processed"
    summary = _load_json(
        processed / "data_prep_summary.json", checker, "data_prep_summary.json"
    )
    if summary is None:
        return

    checker.check(
        summary.get("articles") == EXPECTED_ARTICLES,
        f"{EXPECTED_ARTICLES} articles were kept",
        summary.get("articles"),
    )
    checker.check(
        summary.get("splits") == EXPECTED_SPLITS,
        "train/val/test split sizes are 2700/150/150",
        summary.get("splits"),
    )
    checker.check(
        summary.get("punctuation", {}).get("train", {}).get("examples")
        == EXPECTED_PUNCT_TRAIN_WINDOWS,
        f"{EXPECTED_PUNCT_TRAIN_WINDOWS} punctuation training windows",
        summary.get("punctuation", {}).get("train", {}).get("examples"),
    )
    checker.check(
        summary.get("ordering", {}).get("train", {}).get("pairs")
        == EXPECTED_ORDER_TRAIN_PAIRS,
        f"{EXPECTED_ORDER_TRAIN_PAIRS} ordering training pairs",
        summary.get("ordering", {}).get("train", {}).get("pairs"),
    )

    article_ids = {}
    for split in SPLITS:
        punct_path = processed / "punctuation" / f"{split}.jsonl"
        order_path = processed / "ordering" / f"{split}.jsonl"

        try:
            punct_rows = read_jsonl(punct_path)
        except Exception as error:  # noqa: BLE001
            checker.check(False, f"punctuation/{split}.jsonl is readable", error)
            punct_rows = []

        if punct_rows:
            expected = summary["punctuation"][split]["examples"]
            checker.check(
                len(punct_rows) == expected,
                f"punctuation/{split}.jsonl row count matches the summary",
                f"{len(punct_rows)} != {expected}",
            )
            misaligned = 0
            out_of_range = 0
            bad_labels = 0
            for row in punct_rows:
                words = row.get("words") or []
                labels = row.get("labels") or []
                if len(words) != len(labels):
                    misaligned += 1
                if not (MIN_WORDS <= len(words) <= MAX_WORDS):
                    out_of_range += 1
                if any(label not in LABELS for label in labels):
                    bad_labels += 1
            checker.check(misaligned == 0, f"{split} words and labels align", misaligned)
            checker.check(
                out_of_range == 0,
                f"{split} windows have {MIN_WORDS}-{MAX_WORDS} words",
                out_of_range,
            )
            checker.check(
                bad_labels == 0,
                f"{split} labels are all in {LABELS}",
                bad_labels,
            )
            article_ids[split] = {
                row.get("article_id") for row in punct_rows if row.get("article_id")
            }

        try:
            order_rows = read_jsonl(order_path)
        except Exception as error:  # noqa: BLE001
            checker.check(False, f"ordering/{split}.jsonl is readable", error)
            order_rows = []

        if order_rows:
            expected = summary["ordering"][split]["pairs"]
            checker.check(
                len(order_rows) == expected,
                f"ordering/{split}.jsonl row count matches the summary",
                f"{len(order_rows)} != {expected}",
            )
            positive = sum(1 for row in order_rows if row.get("label") == 1)
            share = positive / len(order_rows)
            checker.check(
                0.48 <= share <= 0.52,
                f"{split} ordering pairs are approximately balanced",
                f"positive share {share:.4f}",
            )
            bad_pairs = sum(
                1
                for row in order_rows
                if not row.get("sentence_a", "").strip()
                or not row.get("sentence_b", "").strip()
                or row.get("label") not in (0, 1)
            )
            checker.check(bad_pairs == 0, f"{split} ordering pairs are valid", bad_pairs)

        eval_path = processed / "ordering" / "ordering_eval.jsonl"
        if split == "test":
            try:
                eval_articles = read_jsonl(eval_path)
            except Exception as error:  # noqa: BLE001
                checker.check(False, "ordering_eval.jsonl is readable", error)
                eval_articles = []
            if eval_articles:
                checker.check(
                    len(eval_articles) == EXPECTED_ORDER_EVAL_ARTICLES,
                    f"{EXPECTED_ORDER_EVAL_ARTICLES} held-out ordering articles",
                    len(eval_articles),
                )
                bad_lengths = sum(
                    1
                    for article in eval_articles
                    if not (8 <= len(article.get("sentences") or []) <= 70)
                )
                checker.check(
                    bad_lengths == 0,
                    "ordering evaluation articles have 8-70 sentences",
                    bad_lengths,
                )

    if len(article_ids) == 3:
        overlap = (
            article_ids["train"] & article_ids["val"]
            or article_ids["train"] & article_ids["test"]
            or article_ids["val"] & article_ids["test"]
        )
        checker.check(not overlap, "no article appears in two splits", len(overlap))


def check_models(root, checker, smoke):
    checker.section("models and training")
    models = Path(root) / "models"
    for name, threshold in (
        ("punctuation", MIN_TRAIN_BOUNDARY_F1),
        ("ordering", MIN_TRAIN_ORDER_ACCURACY),
    ):
        adapter = models / name / "adapter"
        checker.check(
            (adapter / "adapter_config.json").is_file(),
            f"{name} adapter_config.json exists",
            adapter,
        )
        checker.check(
            (adapter / "adapter_model.safetensors").is_file(),
            f"{name} adapter weights exist",
            adapter,
        )
        checker.check(
            (adapter / "tokenizer_config.json").is_file(),
            f"{name} tokenizer files exist",
            adapter,
        )

        summary = _load_json(
            models / name / "training_summary.json",
            checker,
            f"{name}/training_summary.json",
        )
        if summary is None:
            continue
        validation = summary.get("validation_metrics") or {}
        test = summary.get("test_metrics") or {}
        if name == "punctuation":
            value = validation.get("eval_boundary_f1", 0.0)
            checker.check(
                value >= threshold,
                f"punctuation validation boundary F1 >= {threshold}",
                value,
            )
            value = test.get("test_boundary_f1", 0.0)
            checker.check(
                value >= threshold,
                f"punctuation test boundary F1 >= {threshold}",
                value,
            )
        else:
            value = validation.get("eval_accuracy", 0.0)
            checker.check(
                value >= threshold,
                f"ordering validation accuracy >= {threshold}",
                value,
            )
            value = test.get("test_accuracy", 0.0)
            checker.check(
                value >= threshold,
                f"ordering test accuracy >= {threshold}",
                value,
            )

    if smoke:
        smoke_models(root, checker)


def smoke_models(root, checker):
    set_seed(SEED)
    models = Path(root) / "models"
    processed = Path(root) / "data" / "processed"

    try:
        from src.punctuation.predict import PunctuationRestorer

        rows = read_jsonl(processed / "punctuation" / "test.jsonl")
        restorer = PunctuationRestorer(models / "punctuation", memory_gb=1.5)
        try:
            labels, punctuated, plain = restorer.restore(rows[0]["words"])
            first = restorer.predict_labels(rows[0]["words"])
            second = restorer.predict_labels(rows[0]["words"])
        finally:
            restorer.close()
        checker.check(
            len(labels) == len(rows[0]["words"]),
            "punctuation predictions align with the input words",
        )
        checker.check(
            all(0 <= label < len(LABELS) for label in labels),
            "punctuation predictions are valid label ids",
        )
        checker.check(
            len(punctuated) == len(plain) >= 1 and all(item.strip() for item in punctuated),
            "punctuation restoration produces non-empty sentences",
        )
        checker.check(first == second, "punctuation inference is deterministic")
    except Exception as error:  # noqa: BLE001
        checker.check(False, "punctuation adapter smoke test runs", error)

    try:
        from src.ordering.decode import OrderScorer, order_sentences

        articles = read_jsonl(processed / "ordering" / "ordering_eval.jsonl")
        sentences = articles[0]["sentences"][:8]
        scorer = OrderScorer(models / "ordering", memory_gb=1.5)
        try:
            matrix = scorer.score_matrix(sentences)
            orders = {
                strategy: order_sentences(sentences, scorer, strategy=strategy)
                for strategy in ("borda", "clusters", "beam")
            }
        finally:
            scorer.close()
        checker.check(
            matrix.shape == (len(sentences), len(sentences)),
            "ordering score matrix has the expected shape",
            matrix.shape,
        )
        checker.check(
            all(abs(matrix[i, i] - 0.5) < 1e-9 for i in range(len(sentences))),
            "ordering score matrix diagonal is 0.5",
        )
        complement_errors = sum(
            1
            for i in range(len(sentences))
            for j in range(len(sentences))
            if abs(matrix[i, j] + matrix[j, i] - 1.0) >= 1e-6
        )
        checker.check(
            complement_errors == 0,
            "ordering scores are symmetric-complementary",
            complement_errors,
        )
        expected = list(range(len(sentences)))
        checker.check(
            all(sorted(order) == expected for order in orders.values()),
            "all three decoders return valid permutations",
            {key: order for key, order in orders.items()},
        )
    except Exception as error:  # noqa: BLE001
        checker.check(False, "ordering adapter smoke test runs", error)


def check_reports(root, checker, allow_skipped_translation):
    checker.section("evaluation report")
    report = _load_json(
        Path(root) / "reports" / "evaluation.json", checker, "evaluation.json"
    )
    if report is None:
        return

    checker.check(report.get("no_llm") is True, "report records no_llm=true")
    checker.check(
        report.get("memory_limit_gb") == 1.5,
        "report records the 1.5 GB memory cap",
        report.get("memory_limit_gb"),
    )

    punctuation = report.get("punctuation") or {}
    checker.check(
        punctuation.get("status") == "ok",
        "punctuation evaluation completed",
        punctuation.get("status"),
    )
    if punctuation.get("status") == "ok":
        accuracy = punctuation.get("accuracy", 0.0)
        boundary = (punctuation.get("boundary") or {}).get("f1", 0.0)
        per_class = punctuation.get("per_class_f1") or {}
        checker.check(
            accuracy >= MIN_PUNCT_ACCURACY,
            f"punctuation accuracy >= {MIN_PUNCT_ACCURACY}",
            accuracy,
        )
        checker.check(
            boundary >= MIN_BOUNDARY_F1,
            f"punctuation boundary F1 >= {MIN_BOUNDARY_F1}",
            boundary,
        )
        checker.check(
            per_class.get("none", 0.0) >= MIN_NO_PUNCT_F1,
            f"punctuation class 'none' F1 >= {MIN_NO_PUNCT_F1}",
            per_class.get("none"),
        )
        checker.check(
            per_class.get("sentence_end", 0.0) >= MIN_SENTENCE_END_F1,
            f"punctuation class 'sentence_end' F1 >= {MIN_SENTENCE_END_F1}",
            per_class.get("sentence_end"),
        )

    ordering = report.get("ordering") or {}
    checker.check(
        ordering.get("status") == "ok",
        "ordering evaluation completed",
        ordering.get("status"),
    )
    if ordering.get("status") == "ok":
        pairs = ordering.get("pairs") or {}
        accuracy = pairs.get("accuracy", 0.0)
        macro_f1 = pairs.get("macro_f1", 0.0)
        taus = ordering.get("kendall_tau") or {}
        borda = (taus.get("borda") or {}).get("mean_tau", -1.0)
        tfidf = (taus.get("tfidf_ablation") or {}).get("mean_tau", -1.0)
        checker.check(
            accuracy >= MIN_ORDER_ACCURACY,
            f"ordering pair accuracy >= {MIN_ORDER_ACCURACY}",
            accuracy,
        )
        checker.check(
            macro_f1 >= MIN_ORDER_MACRO_F1,
            f"ordering macro F1 >= {MIN_ORDER_MACRO_F1}",
            macro_f1,
        )
        checker.check(
            borda >= MIN_BORDA_TAU,
            f"Borda mean Kendall tau >= {MIN_BORDA_TAU}",
            borda,
        )
        checker.check(
            borda > tfidf,
            "Borda beats the TF-IDF-only ablation",
            f"borda={borda}, tfidf={tfidf}",
        )

    translation = report.get("translation") or {}
    status = translation.get("status")
    if status == "ok":
        checker.check(
            translation.get("examples", 0) > 0,
            "translation evaluation used at least one example",
            translation.get("examples"),
        )
        checker.check(
            translation.get("bleu", 0.0) >= MIN_BLEU,
            f"translation BLEU >= {MIN_BLEU}",
            translation.get("bleu"),
        )
        checker.check(
            translation.get("chrf", 0.0) >= MIN_CHRF,
            f"translation chrF >= {MIN_CHRF}",
            translation.get("chrf"),
        )
    elif allow_skipped_translation:
        print(f"  skip translation evaluation was {status!r} (allowed)")
    else:
        checker.check(
            False,
            "translation evaluation completed (login to the gated IndicTrans2 repo)",
            f"status={status!r}, reason={translation.get('reason')}",
        )


def check_outputs(root, checker, expected_stems):
    checker.section("end-to-end outputs")
    summary = _load_json(
        Path(root) / "reports" / "run_summary.json", checker, "run_summary.json"
    )
    if summary is None:
        return

    checker.check(summary.get("group_size") == 5, "groups contain five sentences")
    checker.check(
        summary.get("ordering_strategy") in ("borda", "clusters", "beam"),
        "a valid ordering strategy was recorded",
        summary.get("ordering_strategy"),
    )
    checker.check(
        summary.get("translation_model") is not None,
        "translation ran (translation_model is not null)",
        summary.get("translation_model"),
    )

    files = summary.get("files") or {}
    actual_stems = sorted(files)
    checker.check(
        actual_stems == sorted(expected_stems),
        "run_summary lists every expected input file",
        f"{actual_stems} != {sorted(expected_stems)}",
    )

    outputs = Path(root) / "outputs"
    for stem in expected_stems:
        info = files.get(stem)
        if not info:
            checker.check(False, f"{stem} appears in run_summary.json")
            continue

        paths = {
            "hi": outputs / f"{stem}.sentences_hi.txt",
            "en": outputs / f"{stem}.sentences_en.txt",
            "ghi": outputs / f"{stem}.groups_hi.txt",
            "gen": outputs / f"{stem}.groups_en.txt",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            checker.check(False, f"{stem} has all four output files", missing)
            continue

        lines = {
            name: path.read_text(encoding="utf-8").splitlines()
            for name, path in paths.items()
        }
        non_empty = all(lines.values())
        checker.check(non_empty, f"{stem} output files are non-empty")

        hi_count = len(lines["hi"])
        en_count = len(lines["en"])
        ghi_count = len(lines["ghi"])
        gen_count = len(lines["gen"])
        expected_groups = math.ceil(hi_count / 5)
        checker.check(
            hi_count == en_count,
            f"{stem} Hindi/English sentence counts match",
            f"{hi_count} != {en_count}",
        )
        checker.check(
            ghi_count == gen_count == expected_groups,
            f"{stem} group counts match ceil(sentences / 5)",
            f"{ghi_count}/{gen_count} != {expected_groups}",
        )
        checker.check(
            all(line.strip() for line in sum(lines.values(), [])),
            f"{stem} has no blank lines",
        )
        checker.check(
            not any("NULL" in line for line in lines["hi"]),
            f"{stem} has no leftover NULL tokens",
        )
        checker.check(
            " ".join(lines["ghi"]) == " ".join(lines["hi"]),
            f"{stem} Hindi groups preserve every sentence",
        )
        checker.check(
            " ".join(lines["gen"]) == " ".join(lines["en"]),
            f"{stem} English groups preserve every sentence",
        )
        checker.check(
            any("\u0900" <= char <= "\u097f" for char in "".join(lines["hi"])),
            f"{stem} Hindi output contains Devanagari",
        )
        checker.check(
            any(char.isascii() and char.isalpha() for char in "".join(lines["en"])),
            f"{stem} English output contains Latin letters",
        )
        checker.check(
            info.get("sentences") == hi_count,
            f"{stem} summary sentence count matches the files",
            f"{info.get('sentences')} != {hi_count}",
        )
        checker.check(
            info.get("groups") == ghi_count,
            f"{stem} summary group count matches the files",
            f"{info.get('groups')} != {ghi_count}",
        )
        checker.check(
            info.get("words", 0) > 0,
            f"{stem} summary records a positive word count",
            info.get("words"),
        )

    timings = summary.get("timings_seconds") or {}
    checker.check(
        all(isinstance(timings.get(key), (int, float)) for key in ("punctuation", "ordering", "translation")),
        "all three stage timings were recorded",
        timings,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-reports", action="store_true")
    parser.add_argument("--skip-outputs", action="store_true")
    parser.add_argument(
        "--skip-model-smoke",
        action="store_true",
        help="Do not load the adapters; only check their files and summaries.",
    )
    parser.add_argument(
        "--allow-skipped-translation",
        action="store_true",
        help="Accept a skipped translation evaluation (not valid for the full run).",
    )
    parser.add_argument(
        "--expected-stems",
        default=",".join(EXPECTED_STEMS),
        help="Comma-separated output stems expected from the pipeline.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    checker = Checker()

    if not args.skip_data:
        check_data(root, checker)
    if not args.skip_models:
        check_models(root, checker, smoke=not args.skip_model_smoke)
    if not args.skip_reports:
        check_reports(root, checker, args.allow_skipped_translation)
    if not args.skip_outputs:
        stems = [stem.strip() for stem in args.expected_stems.split(",") if stem.strip()]
        check_outputs(root, checker, stems)

    print(f"\n{checker.checks} checks, {len(checker.failures)} failed")
    if checker.failures:
        print("\nFailures:")
        for failure in checker.failures:
            print(f"  - {failure}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
