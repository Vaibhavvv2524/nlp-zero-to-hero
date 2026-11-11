# Hindi Text Restoration and Translation

End-to-end solution for the Hindi NLP assignment:

1. **Restore punctuation and sentence boundaries** in unpunctuated, shuffled
   Hindi text.
2. **Recover the sentence order** and group the result into blocks of five.
3. **Translate** the ordered text into English, preserving the groups.

No LLM is used. The only models are **mBERT** (stages 1 and 2, fine-tuned with
LoRA) and **IndicTrans2** (stage 3). There is no fallback translation model.

| Stage | Model | Output |
| --- | --- | --- |
| Punctuation | `bert-base-multilingual-cased` + LoRA | `none`, `comma`, `sentence_end`, `question`, `exclamation` per word |
| Ordering | `bert-base-multilingual-cased` + LoRA | pairwise "A before B" probabilities, decoded into a full order |
| Translation | `ai4bharat/indictrans2-indic-en-dist-200M` | English sentence / group translations |

All stages respect a **1.5 GB GPU memory cap**
(`torch.cuda.set_per_process_memory_fraction`). Models are loaded one at a
time and released before the next stage starts.

## Repository layout

```text
nlp-zero-to-hero/
|-- data/
|   |-- raw/                     # the 7 assignment files (space-separated words, NULL tokens)
|   `-- processed/               # generated datasets (git-ignored)
|-- src/
|   |-- constants.py             # paths, model ids, labels, hyper-parameters
|   |-- utils.py                 # seeding, memory cap, Devanagari-safe punctuation helpers
|   |-- data_prep.py             # Hindi Wikipedia -> punctuation + ordering datasets
|   |-- punctuation/
|   |   |-- dataset.py           # word -> subword label alignment
|   |   |-- train.py             # BERT + LoRA token classification
|   |   `-- predict.py           # sliding-window inference
|   |-- ordering/
|   |   |-- pairs.py             # before/after pair dataset
|   |   |-- train.py             # BERT + LoRA pair classifier
|   |   `-- decode.py            # TF-IDF clustering, Borda, beam, Kendall's tau
|   |-- translation/
|   |   `-- indictrans.py        # IndicTrans2 wrapper
|   |-- pipeline.py              # end-to-end assignment pipeline
|   `-- evaluate.py              # metrics + reports
|-- notebooks/
|   `-- colab_end_to_end.ipynb   # full run on a Colab GPU
|-- scripts/colab_setup.sh       # dependency installer
|-- baseline/                    # earlier rule-based version (kept for comparison)
|-- lessons/                     # early generic NLP exercises
|-- outputs/                     # Hindi/English sentence and group files
|-- reports/                     # JSON metrics
|-- requirements-colab.txt       # GPU dependencies
`-- requirements.txt             # local CPU environment
```

## Environment

The primary local environment is a Python 3.13 virtual environment at `.venv/`
with the core CPU libraries (NumPy, pandas, scikit-learn, SciPy, Jupyter). A
separate Python 3.12 environment was used to verify the full deep-learning
pipeline locally on Apple MPS (`torch 2.15`, `transformers 4.44.2`,
`peft 0.13.2`, `datasets 3.6.0`). For a fresh environment use the pins in
`requirements-colab.txt`; that is the exact combination verified end-to-end.

```bash
source .venv/bin/activate
python setup_check.py            # 28/28 checks
python -m src.data_prep --help   # CPU-only CLIs work locally
python -m src.pipeline --help
python -m src.evaluate --help
```

The Jupyter kernel **Python (NLP Zero to Hero)** (`nlp-zero-to-hero`) is already
registered.

## Colab quickstart

1. Open `notebooks/colab_end_to_end.ipynb` in Colab.
2. Set **Runtime > Change runtime type > GPU (T4)**.
3. Upload the project folder to `/content/nlp-zero-to-hero` (or clone your
   GitHub repo there).
4. Run all cells. The notebook installs dependencies, prepares data, trains
   both adapters, evaluates every stage, runs the pipeline and zips the
   outputs.

The IndicTrans2 checkpoint is gated: accept its licence on Hugging Face and
authenticate (`huggingface-cli login`) before the translation cells run.

Command-line equivalent:

```bash
bash scripts/colab_setup.sh

python -m src.data_prep --num-articles 3000 --max-train-pairs 40000

python -m src.punctuation.train --epochs 3 --batch-size 8 --learning-rate 3e-4
python -m src.ordering.train    --epochs 2 --batch-size 8 --learning-rate 2e-4

python -m src.evaluate
python -m src.pipeline --strategy borda
```

A smaller first run that finishes faster:

```bash
python -m src.data_prep --num-articles 1200 --max-train-pairs 20000
python -m src.punctuation.train --max-train-samples 6000
python -m src.ordering.train    --max-train-samples 15000
```

## Verified local run (Apple M4, MPS)

The full pipeline was executed locally with the pinned dependency set. The
official IndicTrans2 repository is gated, so translation used an ungated merged
mirror placed at `models/translation/indictrans2-indic-en-200M` (the wrapper
accepts any local path or Hub id). Training used subsets because MPS is much
slower than a T4:

```bash
python -m src.data_prep --num-articles 3000 --max-train-pairs 40000
python -m src.punctuation.train --epochs 2 --batch-size 16 \
    --learning-rate 3e-4 --max-train-samples 4000
python -m src.ordering.train --epochs 1 --batch-size 16 \
    --learning-rate 2e-4 --max-train-samples 4000
python -m src.evaluate \
    --translation-model models/translation/indictrans2-indic-en-200M \
    --max-translation-examples 100
python -m src.pipeline \
    --translation-model models/translation/indictrans2-indic-en-200M \
    --translation-batch-size 8
```

| Stage | Metric | Value |
| --- | --- | --- |
| Punctuation | token accuracy / boundary F1 | 0.8724 / 0.8842 |
| Ordering | pair macro F1 / Borda Kendall's tau | 0.6534 / 0.2007 |
| Translation | BLEU / chrF (100 IITB sentences) | 34.125 / 59.686 |

All seven `outputs/sample-N.*` files and the three reports were produced by
these commands. Ordering is the weakest stage with only 2,000 training pairs;
increase `--max-train-samples` (the Colab recipe uses the full 40,000 pairs)
for a higher Kendall's tau.

## How each stage works

### 1. Punctuation restoration

* **Data.** Hindi Wikipedia is streamed with `datasets`. Sentences are parsed
  with `str.split()` only. A `\w` regex is never used because it breaks
  Devanagari matras (`अयोध्या -> अयधय`).
* **Labels.** Every word receives one of five labels. A comma after a word is
  `comma`; `।`, `.`, `?`, `!` become `sentence_end`, `question`,
  `exclamation`.
* **Windows.** 32-96 words with stride 40, and the sentences inside each window
  are shuffled. This matches the shuffled assignment input and prevents the
  model from relying on cross-sentence continuity.
* **Subword alignment.** The label sits on the first subword of a word; other
  subwords get `-100` and are ignored by the loss.
* **Model.** `bert-base-multilingual-cased` with LoRA (`r=16`, `alpha=32`,
  `query`/`value`) and a fully trained classification head. A class-weighted
  cross-entropy helps the rare question/exclamation classes.
* **Inference.** Overlapping 64-word windows; predictions are averaged with a
  triangular weight so edge words count less. Sentence strings are rebuilt
  after every boundary label.

### 2. Sentence ordering

* **Data.** Consecutive Wikipedia sentences produce `(A, B, 1)` and
  `(B, A, 0)` pairs, so classes are balanced.
* **Model.** mBERT pair classifier over `[CLS] A [SEP] B [SEP]`, fine-tuned
  with LoRA.
* **Decoders (`--strategy`).**
  * `borda` (default): sort by the average probability that a sentence comes
    before every other sentence. This is a robust Borda/Kemeny approximation.
  * `clusters`: TF-IDF (char 2-4 grams) + KMeans, order cluster medoids with the
    pair model, then beam-search inside each cluster. This follows the notes
    most closely.
  * `beam`: beam-search a single chain over all sentences.
  * A TF-IDF-only ordering is reported as an ablation.
* **Metric.** Held-out articles are shuffled (fixed seed) and re-ordered;
  Kendall's tau compares the recovered order with the truth.

### 3. Translation

* `ai4bharat/indictrans2-indic-en-dist-200M` (~200M parameters, gated on the
  Hub). The local verification used an ungated merged mirror at
  `models/translation/indictrans2-indic-en-200M`.
* `IndicTransToolkit.IndicProcessor` adds the `hin_Deva` -> `eng_Latn` tags
  and performs preprocessing/postprocessing; the tokenizer itself is called
  without language kwargs.
* `trust_remote_code=True` loads the model's custom `IndicTrans` classes.
* Sentences are translated individually, then grouped in fives so Hindi and
  English groups stay aligned.
* Evaluation: corpus BLEU and chrF on 200 IITB hi-en test sentences
  (optional; skipped gracefully if the dataset is unavailable).

## Outputs

`python -m src.pipeline` writes, for every `data/raw/sample-N.txt`:

| File | Contents |
| --- | --- |
| `outputs/sample-N.sentences_hi.txt` | one punctuated Hindi sentence per line, in recovered order |
| `outputs/sample-N.groups_hi.txt` | one line per group of five Hindi sentences |
| `outputs/sample-N.sentences_en.txt` | English sentence translations |
| `outputs/sample-N.groups_en.txt` | English translations grouped in fives |

Reports:

* `reports/data_prep_summary.json` - dataset sizes and label distribution
* `reports/run_summary.json` - per-file counts, timings, memory cap
* `reports/evaluation.json` - punctuation, ordering and translation metrics
* `models/punctuation/training_summary.json` and
  `models/ordering/training_summary.json` - training metrics and peak memory

## Memory

* Training and inference call `limit_gpu_memory(1.5)` before loading a model.
  On a 15 GB T4 this sets the process fraction to `1.5 / 15 = 0.1`.
* Base weights are fp16 on CUDA (~0.36 GB for mBERT, ~0.4 GB for IndicTrans2).
* LoRA keeps optimizer state tiny; gradient checkpointing reduces activation
  memory.
* If OOM still occurs, reduce `--batch-size` to 4 and keep gradient
  accumulation at 2.
* The CUDA cap is a no-op on Apple MPS; loading one stage at a time still keeps
  local usage around 1-2 GB. On MPS, training uses eager attention
  (`attention_implementation()` in `src/utils.py`) because SDPA with dropout is
  broken in this torch build.

## No-LLM / no-fallback guarantee

* No GPT, Gemini, Claude or any other language model is called.
* The pipeline raises a clear error when a checkpoint or IndicTrans2 is
  missing; it never silently switches to a weaker model.
* Punctuation, order and translation are all produced by the trained models.

## Legacy baseline

`baseline/` preserves the first rule-based attempt (dynamic-programming
segmentation, TF-IDF greedy ordering, OPUS-MT translation) together with its
outputs and `baseline_summary.json`. It is **not** part of the final solution;
it exists only to show the comparison.

## Submission checklist

- [ ] Push this folder to a GitHub repository (public or as required).
- [ ] Share the Colab notebook link with outputs visible.
- [ ] Run all notebook cells top to bottom on a T4 GPU.
- [ ] Attach or commit `outputs/`, `reports/evaluation.json` and
      `reports/run_summary.json`.
- [ ] Confirm `reports/evaluation.json` shows all three stages marked `"ok"`
      (translation evaluation may be `"skipped"` if IITB is unavailable).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `IndicTransToolkit` import/build error | `bash scripts/colab_setup.sh` (installs from source if needed) |
| `peft` / `transformers` API mismatch | `pip install --upgrade -r requirements-colab.txt` (do not upgrade to transformers 5) |
| `401` / gated repo for IndicTrans2 | accept the licence and `huggingface-cli login`, or pass a local merged copy with `--translation-model PATH` |
| CUDA OOM | lower `--batch-size`, keep `--memory-gb 1.5`, restart runtime |
| Wikipedia dataset id changed | update `WIKI_DATASET` / `WIKI_CONFIG` in `src/constants.py` |
| Slow data download | use `--num-articles 1200` for a first run |
