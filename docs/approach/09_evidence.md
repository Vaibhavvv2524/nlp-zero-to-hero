# 16. Reproducibility: exact environment, data, and commands

<div class="decision">

**APPROACH CHOSEN.** Make the run reproducible from a clean checkout with pinned dependencies, a pinned Wikipedia snapshot, one seed, explicit commands, recorded reports, and a verifier. Separate what was actually run from what is merely documented.

</div>

## 16.1 Environment

| Item | Value |
| --- | --- |
| Python (verified build env) | 3.12 |
| PyTorch | 2.15.0 |
| Device used locally | Apple MPS |
| Primary local venv | `.venv`, Python 3.13, no torch |
| Verified build/test venv | `.venv.backup-20261109-124948` |
| CUDA | not available locally |
| HF token | absent; gated upstream repos return 401 |
| Translation weights | local mirror `models/translation/indictrans2-indic-en-200M` |

The primary `.venv` is not sufficient for model work because it has no torch. Use the backup environment for builds, training, evaluation, and verification.

## 16.2 Dependency pins and why they matter

From `requirements-colab.txt`:

| Package | Pin | Reason |
| --- | --- | --- |
| `transformers` | 4.44.2 | transformers 5 removes the `transformers.onnx` import path used by older stacks. |
| `tokenizers` | 0.19.1 | Compatible with transformers 4.44. |
| `peft` | 0.13.2 | Newer PEFT imports `torch.distributed.tensor`, which old torch builds lack. |
| `datasets` | 3.6.0 | Datasets 4/5 require huggingface-hub ≥ 1.31. |
| `huggingface-hub` | `>=0.24,<1.0` | Keeps datasets and transformers on the same hub API. |
| `accelerate` | `>=0.34,<2.0` | Trainer compatibility. |
| `IndicTransToolkit` | 1.1.1 | Required for IndicTrans2 preprocessing. |
| `sacrebleu` | 2.6.0 | BLEU and chrF. |

Additional runtime constraints:

- SDPA with dropout is broken on MPS in this stack; use eager attention on MPS.
- fp16 is CUDA-only here; use fp32 on MPS.
- IndicTransToolkit needs a small import shim for the transformers 4.x/5.x split.

## 16.3 Data provenance

| Item | Value |
| --- | --- |
| Dataset | `wikimedia/wikipedia` |
| Config | `20231101.hi` |
| Articles kept | 3,000 |
| Split | 2,700 train / 150 validation / 150 test |
| Punctuation train windows | 25,000 |
| Ordering train pairs | 40,000 |
| Ordering evaluation articles | 40 |
| Seed | 42 |

The local adapters are subset adapters: 4,000 punctuation windows and 4,000 ordering pairs. The data files still contain the full caps; the training commands cap the examples.

## 16.4 Full command sequence

### 16.4.1 Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-colab.txt
```

On Colab:

```bash
bash scripts/colab_setup.sh
```

### 16.4.2 Data

```bash
python -m src.data_prep \
  --num-articles 3000 \
  --max-train-windows 25000 \
  --max-train-pairs 40000 \
  --eval-articles 40
```

Expected summary: `reports/data_prep_summary.json` plus JSONL files under `data/processed/punctuation` and `data/processed/ordering`.

### 16.4.3 Training

Full default recipe:

```bash
python -m src.punctuation.train \
  --epochs 3 --batch-size 8 --learning-rate 3e-4

python -m src.ordering.train \
  --epochs 2 --batch-size 8 --learning-rate 2e-4
```

Subset recipe used for the checked-in adapters:

```bash
python -m src.punctuation.train \
  --epochs 2 --batch-size 16 --learning-rate 3e-4 \
  --max-train-samples 4000

python -m src.ordering.train \
  --epochs 1 --batch-size 16 --learning-rate 2e-4 \
  --max-train-samples 4000
```

### 16.4.4 Evaluation

```bash
python -m src.evaluate
```

Fast check without translation:

```bash
python -m src.evaluate --skip-translation
```

### 16.4.5 Pipeline

```bash
python -m src.pipeline --strategy borda
```

### 16.4.6 Verification

```bash
python scripts/verify_colab_run.py
```

The full local run passed 150/150 checks; the `--skip-model-smoke` run passed 142 checks.

## 16.5 What is pinned and what is not

| Pinned | Not pinned |
| --- | --- |
| Dataset id and config | Exact upstream file bytes over time |
| Seed | Hardware nondeterminism |
| Dependency versions | CUDA/cuDNN version on other machines |
| Model ids | Gated model revision hash |
| Decoder settings | Human judgement of translation quality |

For a stronger claim, record model revision hashes and add multi-seed runs. The current reproducibility is sufficient to rerun the same design on the same snapshot, not to guarantee bitwise identical weights across hardware.

## 16.6 Evidence files

| File | What it proves |
| --- | --- |
| `reports/data_prep_summary.json` | Split sizes, label counts, window/pair counts |
| `reports/evaluation.json` | Punctuation, ordering, and translation metrics |
| `reports/run_summary.json` | Timings, file counts, strategy, memory limit |
| `models/punctuation/training_summary.json` | Adapter config, trainable params, validation/test metrics |
| `models/ordering/training_summary.json` | Same for ordering |
| `outputs/sample-N.*` | End-to-end output files |
| `scripts/verify_colab_run.py` | Executable structural and threshold checks |

<div class="pagebreak"></div>

# 17. Implementation map: where every claim lives

## 17.1 Repository layout

```text
src/
  constants.py              all shared configuration
  utils.py                  seeds, devices, memory, text, windows, grouping
  data_prep.py              Wikipedia -> punctuation windows + ordering pairs
  evaluate.py               per-stage metrics and report
  pipeline.py               end-to-end orchestration
  punctuation/
    dataset.py              subword alignment
    train.py                weighted Trainer and metrics
    predict.py              sliding-window inference and reconstruction
  ordering/
    pairs.py                pair dataset
    train.py                pair classifier
    decode.py               score matrix and decoders
  translation/
    indictrans.py           IndicTrans2 wrapper
scripts/
  colab_setup.sh            Colab environment setup
  verify_colab_run.py       fail-fast verification
  build_study_guide.py      PDF/HTML study guide
  build_holy_approach.py    this document
  make_approach_diagrams.py SVG diagrams
notebooks/
  colab_end_to_end.ipynb    59-cell beginner course
  pipeline_demo.ipynb       22-cell executed local demo
docs/
  part1.md .. part5.md      study guide chapters
  approach/                 this approach dossier
  approach_assets/          diagrams
  assblasternlp.pdf         beginner study-guide PDF
  holy-approach.pdf         this dossier
```

## 17.2 Constants and utilities

| Symbol | File | Role |
| --- | --- | --- |
| `BASE_MODEL` | `src/constants.py` | shared mBERT checkpoint |
| `TRANSLATION_MODEL` | `src/constants.py` | IndicTrans2 checkpoint |
| `LABELS`, `LABEL2ID`, `ID2LABEL` | `src/constants.py` | punctuation label taxonomy |
| `PUNCTUATION` | `src/constants.py` | label-to-mark mapping |
| `BOUNDARY_LABELS` | `src/constants.py` | boundary set for metrics |
| window and length constants | `src/constants.py` | data and tokenizer bounds |
| `set_seed` | `src/utils.py` | Python/NumPy/PyTorch seeding |
| `get_device` | `src/utils.py` | CUDA > MPS > CPU |
| `attention_implementation` | `src/utils.py` | eager attention on MPS |
| `limit_gpu_memory` | `src/utils.py` | CUDA process fraction cap |
| `free_memory` | `src/utils.py` | gc + CUDA cache reset |
| `strip_punctuation` | `src/utils.py` | reversible mark extraction |
| `restore_punctuation` | `src/utils.py` | mark reattachment |
| `split_into_sentences` | `src/utils.py` | boundary-based sentence rebuild |
| `sliding_windows`, `window_weight` | `src/utils.py` | punctuation inference layout |
| `group_lines` | `src/utils.py` | chunks of five |

## 17.3 Data preparation

| Function | File | Role |
| --- | --- | --- |
| `parse_article` | `src/data_prep.py` | clean, split, filter sentences |
| `build_punctuation_examples` | `src/data_prep.py` | overlapping shuffled windows |
| `make_pairs_from_sentences` | `src/ordering/pairs.py` | consecutive pairs + reversals |
| `article_split` | `src/data_prep.py` | seeded article-level split |
| `main` | `src/data_prep.py` | writes JSONL and summary |

## 17.4 Stage 1

| Symbol | File | Role |
| --- | --- | --- |
| `PunctuationDataset` | `src/punctuation/dataset.py` | tokenization and first-subword labels |
| `class_weights` | `src/punctuation/train.py` | capped inverse frequency |
| `WeightedTrainer` | `src/punctuation/train.py` | weighted CE with ignore index |
| `compute_metrics` | `src/punctuation/train.py` | accuracy, boundary P/R/F1, per-class F1 |
| `PunctuationRestorer` | `src/punctuation/predict.py` | adapter load and sliding-window inference |
| `predict_labels` | `src/punctuation/predict.py` | aggregated per-word labels |
| `restore` | `src/punctuation/predict.py` | labels to punctuated/plain sentences |

## 17.5 Stage 2

| Symbol | File | Role |
| --- | --- | --- |
| `SentencePairDataset` | `src/ordering/pairs.py` | pair tokenization |
| `compute_metrics` | `src/ordering/train.py` | pair accuracy and macro F1 |
| `OrderScorer` | `src/ordering/decode.py` | adapter load and symmetric score matrix |
| `score_matrix` | `src/ordering/decode.py` | one score per unordered pair |
| `_earliness` | `src/ordering/decode.py` | mean before-score |
| `order_from_matrix` | `src/ordering/decode.py` | decoder dispatch |
| `_beam_order` | `src/ordering/decode.py` | beam chain |
| `cluster_sentences` | `src/ordering/decode.py` | TF-IDF + KMeans + medoids |
| `order_sentences_tfidf` | `src/ordering/decode.py` | lexical ablation |
| `kendall_tau`, `evaluate_orderings` | `src/ordering/decode.py` | order evaluation |

## 17.6 Stage 3 and orchestration

| Symbol | File | Role |
| --- | --- | --- |
| `_import_indic_processor` | `src/translation/indictrans.py` | toolkit import compatibility |
| `IndicTrans2Translator` | `src/translation/indictrans.py` | load, translate, close |
| `translate` | `src/translation/indictrans.py` | batch preprocess/generate/postprocess |
| `read_tokens` | `src/pipeline.py` | whitespace tokens minus `NULL` |
| `run` | `src/pipeline.py` | stage orchestration and writing |
| `evaluate_translation` | `src/evaluate.py` | IITB BLEU/chrF and samples |

## 17.7 Claim-to-code mapping

| Claim | Code / evidence |
| --- | --- |
| "No LLM is used" | only `bert-base-multilingual-cased` and `indictrans2-indic-en-dist-200M`; `no_llm: true` |
| "Punctuation preserves words" | `restore_punctuation` appends marks; no generation |
| "Ordering permutes sentences" | `order_sentences` returns indices; pipeline applies permutation |
| "Translation keeps alignment" | one output per input; flat list sliced by counts |
| "Memory stays within one model" | `pipeline.run` loads/closes each stage |
| "Article-level splits" | seeded article split in `data_prep` |
| "Borda is the measured best decoder" | `evaluation.json` tau table |
| "Rare classes are weak" | per-class F1 in `evaluation.json` |

<div class="pagebreak"></div>

# 18. Runbook: operate, inspect, and troubleshoot

## 18.1 Quick local run

```bash
source .venv.backup-20261109-124948/bin/activate
python -m src.pipeline --limit 1 --no-translate
```

Inspect `outputs/sample-1.sentences_hi.txt`. This exercises punctuation and ordering without loading the translation model.

## 18.2 Full local run

```bash
source .venv.backup-20261109-124948/bin/activate
python -m src.pipeline --strategy borda
python scripts/verify_colab_run.py
```

Expected timings from the recorded run:

| Stage | Seconds |
| --- | ---: |
| Punctuation | 2.31 |
| Ordering | 130.53 |
| Translation | 102.17 |
| Total | about 235 |

Expected per-file sentence counts:

| File | Words | `NULL` removed | Sentences | Groups |
| --- | ---: | ---: | ---: | ---: |
| sample-1 | 747 | 5 | 52 | 11 |
| sample-2 | 768 | 5 | 55 | 11 |
| sample-3 | 742 | 6 | 49 | 10 |
| sample-4 | 1,118 | 4 | 53 | 11 |
| sample-5 | 1,026 | 9 | 55 | 11 |
| sample-6 | 507 | 11 | 80 | 16 |
| sample-7 | 846 | 27 | 99 | 20 |

## 18.3 What to inspect manually

| File | What to look for |
| --- | --- |
| `sample-1`–`sample-5` | Prose-like input; check sentence endings and ordering plausibility. |
| `sample-6`, `sample-7` | Dialogue-like input; expect over-segmentation; compare with raw text. |
| `.groups_hi.txt` | Groups should be consecutive sentences, last group possibly shorter. |
| `.sentences_en.txt` | Check alignment, not just fluency. |
| `evaluation.json` | Check counts and per-class F1 before quoting a metric. |
| `run_summary.json` | Check strategy and timings for the run you are quoting. |

## 18.4 Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: torch` | Wrong venv | Use the backup environment. |
| MPS dropout error | SDPA + dropout on MPS | Eager attention is already selected; check the device branch. |
| `transformers.onnx` import error | transformers 5 installed | Install transformers 4.44.2. |
| `torch.distributed.tensor` import error | PEFT too new for torch | Use peft 0.13.2. |
| `huggingface-hub` version conflict | datasets 4/5 | Pin datasets 3.6.0 and hub <1.0. |
| 401 from Hugging Face | Gated model, no token | Use the local IndicTrans2 mirror. |
| `IndicProcessor` import error | Toolkit/transformers path split | Use the shim in `indictrans.py`; install IndicTransToolkit. |
| Missing adapter | Training not run | Run the training command printed in the error. |
| Memory cap reports null | MPS/CPU | Expected; the cap is CUDA-only. |
| Translation slow | fp32 on MPS and four beams | Expected; reduce beams or batch only for demos. |
| Ordering slow | Quadratic pair count | Expected; use `--limit` or a smaller article. |
| PDF build fails | Chrome missing or chart not ready | Check Chrome path and virtual time budget. |
| Notebook mismatch | Edited notebook not re-executed | Re-run cells and save outputs. |

## 18.5 Reproducing the documentation PDFs

Beginner study guide:

```bash
.venv.backup-20261109-124948/bin/python scripts/build_study_guide.py
```

Approach dossier:

```bash
.venv.backup-20261109-124948/bin/python scripts/build_holy_approach.py
```

Diagrams:

```bash
.venv.backup-20261109-124948/bin/python scripts/make_approach_diagrams.py
```

PDFs are local artifacts and are git-ignored via `docs/*.pdf`. Do not commit or push them.

<div class="pagebreak"></div>

# 19. Review: questions, answers, and defence

## 19.1 Why not use an LLM?

The assignment forbids it, and an LLM would make the pipeline opaque, non-deterministic, and hard to evaluate by stage. The system uses mBERT for classification and IndicTrans2 for translation; neither is used as a general reasoning engine.

## 19.2 Why three stages instead of one model?

There are no aligned triplets for end-to-end training. Decomposition gives exact word preservation, localisable errors, and small models that fit the budget. The cost is error compounding, which is measured and documented.

## 19.3 Why is comma F1 so low?

Comma placement is style-dependent and often optional. The training distribution also has very few commas relative to `none` (3.264% vs 90.947%). The model learns some comma cues, but not enough to be reliable. Question and exclamation are worse because there are only 236 and 153 examples.

## 19.4 Why is ordering tau only 0.20?

The pair classifier is only 65.43% accurate, and local pair errors do not compose into a consistent permutation. Borda is the best decoder tested, but no decoder can recover an order the scorer does not know. This is the weakest stage.

## 19.5 Why is Borda better than beam or clusters?

Measured mean tau: Borda 0.2007, clusters 0.1220, beam 0.0921, TF-IDF 0.0631. Borda aggregates all pair evidence and is robust to individual errors. Beam locks in early mistakes; clusters assume topical adjacency.

## 19.6 Why does pair accuracy not equal order accuracy?

Pair accuracy is measured on isolated balanced pairs. A permutation requires transitive consistency. Cycles such as $A \prec B \prec C \prec A$ make a perfect global order impossible even when most pairs are individually correct.

## 19.7 Why train on Wikipedia?

It provides natural punctuation and natural sentence order without synthetic labels or an LLM. The cost is domain mismatch: dialogue and informal text are under-represented, which is visible in sample-6 and sample-7.

## 19.8 Why article-level splits?

Overlapping windows and sentence pairs from the same article would leak document context. Splitting by article ensures no test article shares text with training.

## 19.9 Why LoRA?

Full fine-tuning would use more memory and produce large checkpoints. LoRA trains about 0.33% of parameters and still learns useful task adapters. The trade-off is a slightly lower ceiling.

## 19.10 Why mBERT instead of an Indic-specific encoder?

mBERT is a strong, well-supported multilingual baseline that fits the budget and runs on both Colab and Apple Silicon. MuRIL and IndicBERT are plausible alternatives, but they change the tooling and memory profile and were not measured here.

## 19.11 How do you know the 1.5GB cap is respected?

The design loads one model at a time, and the CUDA fraction cap is implemented in `limit_gpu_memory`. Locally, the cap is a no-op on MPS and was not exercised on CUDA. This is stated explicitly; the project does not claim a measured CUDA peak.

## 19.12 Are the reported scores from the full training recipe?

No. The checked-in adapters are 4,000-example subsets. The default recipe is 25,000 punctuation windows and 40,000 ordering pairs. The document separates subset measurements from the full recipe everywhere it matters.

## 19.13 How is translation evaluated?

On 100 Hindi–English pairs from `cfilt/iitb-english-hindi`, using corpus BLEU and chrF. The assignment files have no reference translations, so their translation quality is inspected qualitatively. BLEU 34.125 is not an end-to-end score.

## 19.14 What is the biggest weakness?

Global ordering. The pair model is modest, the decoder is approximate, and the end-to-end order is only weakly recovered. Punctuation handles sentence endings well but commas poorly; translation is usable but drifts.

## 19.15 What would you improve first?

Train ordering on the full 40,000 pairs with hard negatives, then add a listwise or ILP decoder for global consistency. For punctuation, add dialogue/news data and rare-class diversity rather than more class weighting. For translation, add paragraph context and a glossary.

## 19.16 What is the honest one-sentence claim?

A constrained, inspectable, no-LLM Hindi→English pipeline that restores sentence boundaries well, restores commas modestly, improves order over a lexical baseline without fully recovering document order, and translates at usable quality within a small-model budget.

## 19.17 What should a reviewer verify first?

1. `reports/evaluation.json` for the scoreboard and sample counts.
2. `models/*/training_summary.json` for adapter configs and trainable parameters.
3. `reports/run_summary.json` for timings and output counts.
4. `outputs/sample-6` and `sample-7` for over-segmentation.
5. `scripts/verify_colab_run.py` for the structural checks.
6. `src/constants.py` for the no-LLM model ids and memory cap.

<div class="pagebreak"></div>
