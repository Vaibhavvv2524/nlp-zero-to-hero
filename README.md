# Hindi Text Restoration and Translation

**Repository:** https://github.com/Vaibhavvv2524/nlp-zero-to-hero

**Full Colab build (self-cloning):** https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/colab_end_to_end.ipynb

**Executed demo (read-only, outputs visible):** https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/pipeline_demo.ipynb

> **Reviewers:** open the **executed demo** to see real outputs without running
> anything. The demo loads local model artifacts that are intentionally not
> committed, so do not click **Run all** there. Use the **full Colab build** to
> run the project from scratch; it clones the repository automatically.

This project takes Hindi text that has lost its punctuation and whose sentences
are in the wrong order, then:

1. **restores punctuation and sentence boundaries**,
2. **recovers a sensible sentence order**,
3. **translates the result into English** and groups it in fives.

It is a three-stage NLP pipeline. No LLM or chatbot API is used. The only models
are **mBERT** (stages 1 and 2, fine-tuned with LoRA) and **IndicTrans2**
(stage 3). There is no fallback translation model.

## At a glance

| Stage | Plain-English job | Model | Output |
| --- | --- | --- | --- |
| 1. Punctuation | Decide whether each word is followed by a comma, sentence end, question mark or exclamation mark. | `bert-base-multilingual-cased` + LoRA | Five labels per word, then rebuilt sentences |
| 2. Ordering | Decide which sentence comes before which, then build one order for the whole article. | `bert-base-multilingual-cased` + LoRA pair classifier | Reordered sentences |
| 3. Translation | Turn each Hindi sentence into English without breaking the order or groups. | `ai4bharat/indictrans2-indic-en-dist-200M` | English sentences and groups of five |

All stages respect a **1.5 GB GPU memory cap**. Models are loaded one at a time
and released before the next stage starts.

## New here? Choose a path

- **I just want to see what it produces.** Open the already-executed
  `notebooks/pipeline_demo.ipynb`, or look at `outputs/` and `reports/`.
- **I want to run the full pipeline.** Follow the
  [Colab quickstart](#colab-quickstart) below.
- **I want to understand the code.** Read
  [How to read this codebase](#how-to-read-this-codebase) and then
  [How the pipeline works](#how-the-pipeline-works).

You do **not** need to understand every model to run the project. Start with the
executed demo, then read the code one stage at a time.

## Learn the project step by step

There are two notebooks, and they serve different purposes:

| Notebook | Time | What it teaches |
| --- | --- | --- |
| `notebooks/pipeline_demo.ipynb` | ~2-3 min | The executed evidence notebook. Read the saved predictions and output files without running anything. It is **not** a self-contained Colab run. |
| `notebooks/colab_end_to_end.ipynb` | ~30-60 min | The full build. It clones the repo, checks the GPU, installs dependencies, and explains tokens, labels, mBERT, LoRA, training, evaluation and the pipeline one step at a time. |

**Recommended path if this is your first NLP project:**

1. Open `pipeline_demo.ipynb` and read the saved output of each stage.
2. Open `colab_end_to_end.ipynb` and run it from top to bottom on a T4 GPU. The first code cell clones the repository automatically.
3. Use the [glossary](#glossary) whenever a term is unfamiliar.
4. Read the code in the order listed in
   [How to read this codebase](#how-to-read-this-codebase).

## What you give it and what you get back

```text
Input:  Hindi words, no punctuation, sentences shuffled, optional NULL tokens
        (data/raw/sample-N.txt)

Stage 1: add punctuation + sentence boundaries
        "भारत एक विशाल देश है। यह एक परीक्षण है।"

Stage 2: put the sentences back in a likely order

Stage 3: translate to English and group every five sentences

Output: sample-N.sentences_hi.txt, sample-N.groups_hi.txt,
        sample-N.sentences_en.txt, sample-N.groups_en.txt
```

## Colab quickstart

1. Open the [full Colab build](https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/colab_end_to_end.ipynb).
2. Set **Runtime > Change runtime type > GPU (T4)**.
3. Press **Runtime > Run all**. The first code cell clones the repository to
   `/content/nlp-zero-to-hero` automatically. No manual upload is needed.
4. Optional, for the final translation stage: accept the IndicTrans2 licence on
   Hugging Face and add a Colab secret named `HF_TOKEN`. Training cells work
   without it; evaluation and the pipeline need it because the checkpoint is
   gated.
5. The notebook installs dependencies, prepares data, trains both adapters,
   evaluates every stage, runs the pipeline and zips the outputs.
6. Let the self-check cells run. `scripts/verify_colab_run.py` is called after
   data preparation, after evaluation and after the pipeline. It exits non-zero
   on missing files, bad metrics, broken adapters or misaligned outputs.

Command-line equivalent:

```bash
bash scripts/colab_setup.sh

python -m src.data_prep --num-articles 3000 --max-train-pairs 40000

python -m src.punctuation.train --epochs 3 --batch-size 8 --learning-rate 3e-4
python -m src.ordering.train    --epochs 2 --batch-size 8 --learning-rate 2e-4

python -m src.evaluate
python -m src.pipeline --strategy borda
python scripts/verify_colab_run.py
```

A smaller first run that finishes faster:

```bash
python -m src.data_prep --num-articles 1200 --max-train-pairs 20000
python -m src.punctuation.train --max-train-samples 6000
python -m src.ordering.train    --max-train-samples 15000
```

## How to read this codebase

Read in this order. Each step is useful on its own, so you can stop whenever you
have what you need.

1. **`notebooks/pipeline_demo.ipynb`** — the fastest way to see inputs, model
   outputs and final artifacts. It is already executed.
2. **`src/pipeline.py`** — the whole project in one place. It loads each model,
   runs the three stages and writes the output files.
3. **`src/constants.py`** — paths, model names, labels and settings.
4. **`src/punctuation/`** — stage 1. Start with `predict.py` if you want to see
   how words become sentences; open `train.py` when you want the training loop.
5. **`src/ordering/`** — stage 2. `pairs.py` builds "A before B" examples;
   `decode.py` turns pairwise scores into one order.
6. **`src/translation/indictrans.py`** — stage 3. A wrapper around IndicTrans2.
7. **`src/data_prep.py`** and **`src/evaluate.py`** — how training data and
   metrics are made.

Skip `baseline/` and `lessons/` on your first pass. They are historical
material, not part of the final pipeline.

## How the pipeline works

### Stage 1 — punctuation restoration

**The problem.** The input is a bag of Hindi words with no punctuation, and the
sentences are shuffled.

**What the model does.** It looks at each word and predicts one of five labels:
`none`, `comma`, `sentence_end`, `question` or `exclamation`. A `sentence_end`
label is where a new sentence starts.

**How it is trained.**

- Hindi Wikipedia is streamed with `datasets`.
- Sentences are parsed with `str.split()` only. A `\w` regex is never used
  because it breaks Devanagari matras (`अयोध्या -> अयधय`).
- Windows are 32–96 words with stride 40. Sentences inside a window are
  shuffled, matching the assignment input.
- Each word gets one label. The label sits on the **first subword**; the other
  subwords get `-100` and are ignored by the loss.
- The model is mBERT with LoRA (`r=16`, `alpha=32`, targets `query`/`value`) and
  a fully trained classification head. Class-weighted loss helps the rare
  question/exclamation labels.

**At inference.** Overlapping 64-word windows are scored; predictions are
averaged with a triangular weight so edge words count less. The sentence strings
are rebuilt after every boundary label.

### Stage 2 — sentence ordering

**The problem.** The sentences are in the wrong order, and the model must put
them back.

**What the model does.** It reads two sentences and predicts whether A comes
before B. It is trained on consecutive Wikipedia sentences: `(A, B, 1)` and the
swapped pair `(B, A, 0)`, so the classes are balanced.

**Turning pairs into an order.** Three decoders are available with
`--strategy`:

- `borda` (default): score each sentence by its average probability of coming
  before every other sentence, then sort. This is a robust Borda/Kemeny
  approximation.
- `clusters`: TF-IDF (character 2–4 grams) + KMeans, order the cluster medoids
  with the pair model, then beam-search inside each cluster. This follows the
  assignment notes most closely.
- `beam`: one beam-search chain over all sentences.

A TF-IDF-only ordering is reported as an ablation.

**How it is measured.** Held-out articles are shuffled with a fixed seed and
re-ordered. **Kendall's tau** compares the recovered order with the true order.

### Stage 3 — translation

- `ai4bharat/indictrans2-indic-en-dist-200M` (~200M parameters, gated on the
  Hub). The local verification used an ungated merged mirror at
  `models/translation/indictrans2-indic-en-200M`.
- `IndicTransToolkit.IndicProcessor` adds the `hin_Deva` → `eng_Latn` tags and
  handles preprocessing/postprocessing. The tokenizer is called without language
  kwargs.
- `trust_remote_code=True` loads the model's custom `IndicTrans` classes.
- Sentences are translated individually, then grouped in fives so Hindi and
  English groups stay aligned.
- Evaluation uses corpus BLEU and chrF on 200 IITB hi–en test sentences. It is
  optional and is skipped gracefully if the dataset is unavailable.

## Outputs

`python -m src.pipeline` writes four files for every `data/raw/sample-N.txt`:

| File | Contents |
| --- | --- |
| `outputs/sample-N.sentences_hi.txt` | One punctuated Hindi sentence per line, in recovered order |
| `outputs/sample-N.groups_hi.txt` | One line per group of five Hindi sentences |
| `outputs/sample-N.sentences_en.txt` | English sentence translations |
| `outputs/sample-N.groups_en.txt` | English translations grouped in fives |

Reports:

- `reports/data_prep_summary.json` — dataset sizes and label distribution
- `reports/run_summary.json` — per-file counts, timings, memory cap
- `reports/evaluation.json` — punctuation, ordering and translation metrics
- `models/punctuation/training_summary.json` and
  `models/ordering/training_summary.json` — training metrics and peak memory

## Local setup

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

All seven `outputs/sample-N.*` files and the reports were produced by these
commands. Ordering is the weakest stage with only 2,000 training pairs; increase
`--max-train-samples` (the Colab recipe uses the full 40,000 pairs) for a higher
Kendall's tau.

## Memory

- Training and inference call `limit_gpu_memory(1.5)` before loading a model. On
  a 15 GB T4 this sets the process fraction to `1.5 / 15 = 0.1`.
- Base weights are fp16 on CUDA (~0.36 GB for mBERT, ~0.4 GB for IndicTrans2).
- LoRA keeps optimizer state tiny; gradient checkpointing reduces activation
  memory.
- If OOM still occurs, reduce `--batch-size` to 4 and keep gradient accumulation
  at 2.
- The CUDA cap is a no-op on Apple MPS; loading one stage at a time still keeps
  local usage around 1–2 GB. On MPS, training uses eager attention
  (`attention_implementation()` in `src/utils.py`) because SDPA with dropout is
  broken in this torch build.

## Project map

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
|   |-- colab_end_to_end.ipynb   # full run on a Colab GPU
|   `-- pipeline_demo.ipynb      # executed local demo with visible outputs
|-- scripts/
|   |-- colab_setup.sh           # dependency installer
|   `-- verify_colab_run.py      # fail-fast checks for a full run
|-- baseline/                    # earlier rule-based version (kept for comparison)
|-- lessons/                     # early generic NLP exercises
|-- outputs/                     # Hindi/English sentence and group files
|-- reports/                     # JSON metrics
|-- requirements-colab.txt       # GPU dependencies
`-- requirements.txt             # local CPU environment
```

## Glossary

| Term | Meaning here |
| --- | --- |
| **Token / subword** | A piece of a word produced by the tokenizer. One Hindi word can become several subwords. |
| **Label** | The punctuation class assigned to a word: `none`, `comma`, `sentence_end`, `question` or `exclamation`. |
| **mBERT** | `bert-base-multilingual-cased`, a multilingual Transformer that understands Hindi text. |
| **LoRA** | A small set of trainable weights added to a frozen model. It makes fine-tuning cheap. |
| **Adapter** | The saved LoRA weights in `models/*/adapter/`. |
| **Pair classifier** | A model that reads two sentences and decides whether the first comes before the second. |
| **Decoder** | The rule that turns many pairwise scores into one complete sentence order (`borda`, `clusters` or `beam`). |
| **Borda** | A voting-style method: each sentence gets an average "comes before others" score, then sentences are sorted. |
| **Kendall's tau** | A score from -1 to 1 measuring how similar two orderings are. Higher is better. |
| **BLEU / chrF** | Standard machine-translation scores. BLEU rewards matching word sequences; chrF works at the character level. |
| **JSONL** | A file with one JSON object per line, used for the datasets. |

## Important design choices

- **No LLM.** No GPT, Gemini, Claude or other language model is called.
- **No silent fallback.** The pipeline raises a clear error when a checkpoint or
  IndicTrans2 is missing; it never switches to a weaker model.
- **Memory cap.** The 1.5 GB CUDA limit is enforced with
  `torch.cuda.set_per_process_memory_fraction`, so exceeding it fails instead
  of silently using more.
- **One stage at a time.** Each model is released before the next one loads.

## Legacy baseline

`baseline/` preserves the first rule-based attempt (dynamic-programming
segmentation, TF-IDF greedy ordering, OPUS-MT translation) together with its
outputs and `baseline_summary.json`. It is **not** part of the final solution;
it exists only to show the comparison.

## Submission checklist

- [x] Push this folder to a GitHub repository:
      https://github.com/Vaibhavvv2524/nlp-zero-to-hero
- [x] Share the self-cloning full-build Colab link.
- [x] Share the executed demo notebook with visible outputs.
- [ ] Optional: run the full build top to bottom on a T4. It is documented and
      self-cloning, but the saved evidence comes from the local demo run.
- [x] Commit `outputs/`, `reports/evaluation.json` and
      `reports/run_summary.json`.
- [x] Confirm `reports/evaluation.json` shows all three stages marked `"ok"`
      (translation evaluation may be `"skipped"` if IITB is unavailable).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `IndicTransToolkit` import/build error | `bash scripts/colab_setup.sh` (installs from source if needed) |
| `peft` / `transformers` API mismatch | `pip install --upgrade -r requirements-colab.txt` (do not upgrade to transformers 5) |
| `401` / gated repo for IndicTrans2 | Accept the licence and `huggingface-cli login`, or pass a local merged copy with `--translation-model PATH` |
| CUDA OOM | Lower `--batch-size`, keep `--memory-gb 1.5`, restart runtime |
| Wikipedia dataset id changed | Update `WIKI_DATASET` / `WIKI_CONFIG` in `src/constants.py` |
| Slow data download | Use `--num-articles 1200` for a first run |
| Notebook self-check failed | Read the failure list printed by `scripts/verify_colab_run.py`; it names the missing file or metric threshold |
