# Legacy baseline (kept for comparison only)

This folder contains the first, rule-based implementation built before the
trained pipeline. It is **not** used by the final assignment.

| File | What it does |
| --- | --- |
| `hindi_utils.py` | Unicode-safe `str.split` tokenizer that removes `NULL` tokens |
| `segment.py` | Dynamic-programming sentence splitter using verb endings, starters, `NULL` hints and a length prior |
| `order.py` | TF-IDF (char 2-4 grams) + KMeans + greedy ordering, with a synthetic Kendall's tau check |
| `translate.py` | `Helsinki-NLP/opus-mt-hi-en` MarianMT wrapper |
| `pipeline.py` | End-to-end rule baseline |
| `baseline_outputs/` | Outputs produced by the baseline |
| `baseline_summary.json` | Baseline run statistics |

Run it from the repository root:

```bash
python -m baseline.pipeline              # segmentation + ordering (CPU only)
python -m baseline.pipeline --translate  # also translate (needs torch)
```

The final pipeline in `src/` replaces every heuristic with a trained model,
as required by the assignment.
