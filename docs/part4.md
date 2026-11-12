## 21. Pipeline orchestration

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Orchestrate with a thin, explicit script rather than a framework. Load all text first, then run stage 1 on every file, free it, run stage 2 on every file, free it, run stage 3 on all sentences, and write aligned outputs. The code is short enough to audit and makes the memory story obvious.
</div>

`src/pipeline.py` is the glue. It is short, and every line matters.

### 21.1 Reading tokens

```python
def read_tokens(path):
    text = Path(path).read_text(encoding='utf-8')
    raw_tokens = text.split()
    words = [token for token in raw_tokens if token != 'NULL']
    return words, len(raw_tokens) - len(words)
```

- `str.split()` handles arbitrary whitespace and preserves Devanagari.
- `NULL` tokens are removed. The count is recorded for the report.
- All records are loaded before any model runs, so the memory used by raw text is negligible.

### 21.2 Stage 1

```python
restorer = PunctuationRestorer(args.punctuation_model, memory_gb=args.memory_gb)
for stem, record in records.items():
    labels, sentences, plain_sentences = restorer.restore(record['words'])
    ...
restorer.close()
```

The restorer returns three things:

- `labels`: one label ID per word,
- `sentences`: punctuated Hindi sentences,
- `plain_sentences`: the same sentences without punctuation, for the ordering model.

### 21.3 Stage 2

```python
scorer = OrderScorer(args.ordering_model, memory_gb=args.memory_gb)
for stem, record in records.items():
    order = order_sentences(record['sentences_plain'], scorer,
                            strategy=args.strategy, ...)
    record['ordered_hi'] = [record['sentences_hi'][i] for i in order]
    record['groups_hi'] = group_lines(record['ordered_hi'], GROUP_SIZE)
scorer.close()
```

Note the index mapping: the order is computed on **plain** sentences, but applied to the **punctuated** sentences. The two lists have the same length and the same sentence boundaries, so the mapping is valid. This is a small but important detail: the ordering model sees the text distribution it was trained on, while the output keeps punctuation.

### 21.4 Stage 3

```python
flat_sentences = [s for record in records.values() for s in record['ordered_hi']]
translations = translator.translate(flat_sentences,
                                    batch_size=args.translation_batch_size)
```

All sentences from all files are flattened into one list and translated in batches. Then the translations are sliced back per file using the sentence counts. This maximizes batch utilization.

Grouping:

```python
record['groups_en'] = [
    ' '.join(group)
    for group in group_lines(record['ordered_en'], GROUP_SIZE)
]
```

The Hindi groups are built from sentences, the English groups from translations, and both use the same `GROUP_SIZE=5`, so they stay aligned.

### 21.5 Saving

For each file, four outputs are written. The summary records words, `NULL` removed, sentence count, group count, and stage timings. It is saved to `reports/run_summary.json`.

### 21.6 Failure behavior

- Missing punctuation adapter: `FileNotFoundError` with a message telling you to train it.
- Missing ordering adapter: same.
- Missing IndicTrans2 or toolkit: `ImportError` with a setup hint.
- No input files: `SystemExit` with the directory path.
- Translation failure: raises; there is no fallback model.

This is deliberate. Silent fallbacks make results hard to reproduce and can hide a broken setup.

### 21.7 CLI options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-dir` | `data/raw` | where the `.txt` inputs live |
| `--output-dir` | `outputs` | where the four files per input are written |
| `--punctuation-model` | `models/punctuation` | adapter directory |
| `--ordering-model` | `models/ordering` | adapter directory |
| `--translation-model` | official Hub id | local path or Hub id |
| `--limit` | none | process only the first N files |
| `--strategy` | `borda` | `borda`, `clusters`, `beam` |
| `--n-clusters` | auto | override cluster count |
| `--beam-width` | 8 | beam width for beam/cluster decoding |
| `--translation-batch-size` | 4 | batch size for translation |
| `--memory-gb` | 1.5 | CUDA memory cap |
| `--no-translate` | off | skip stage 3 |

> **Check yourself.** Why load all records before running any model? *Because the models are the memory bottleneck, not the text. Loading the text first lets the pipeline process every file with one model load, instead of reloading the model per file.*

## 22. Evaluation

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Evaluate each stage on its own held-out data with the metric that matches the task, and write a single JSON report. Optional stages skip gracefully when data or models are unavailable. The evaluation uses the same inference code as the pipeline, so the numbers reflect deployment behavior.
</div>

`src/evaluate.py` measures each stage independently and writes `reports/evaluation.json`.

### 22.1 Punctuation evaluation

- Load 1,500 test windows (108,898 words).
- Run the full sliding-window inference (`PunctuationRestorer.predict_labels`).
- Flatten true and predicted labels.
- Compute token accuracy, boundary precision/recall/F1, and per-class F1.

This is the same inference path used by the pipeline, so the score reflects deployment behavior.

### 22.2 Ordering evaluation

- Load 3,000 balanced test pairs.
- Predict labels with `OrderScorer.predict_pair_labels`.
- Compute accuracy and macro precision/recall/F1.
- Load 40 held-out articles, shuffle each, score all pairs, decode with all strategies plus TF-IDF, and compute Kendall's tau.

### 22.3 Translation evaluation

- Stream `cfilt/iitb-english-hindi` test split.
- Take the first 100 (or `--max-translation-examples`) Hindi/English pairs.
- Translate the Hindi with the same wrapper.
- Compute corpus BLEU and chrF with `sacrebleu`.
- Save five sample translations in the report.

If the dataset cannot be downloaded or `sacrebleu` is missing, the stage is marked `'skipped'` with a reason instead of failing the whole evaluation.

### 22.4 Report structure

```json
{
  'no_llm': true,
  'memory_limit_gb': 1.5,
  'punctuation': {'status': 'ok', '...': '...'},
  'ordering': {'status': 'ok', '...': '...'},
  'translation': {'status': 'ok', '...': '...'}
}
```

The `no_llm` flag is a deliberate assertion: the evaluation itself uses no LLM.

### 22.5 Running subsets

You can skip stages or cap examples:

```bash
python -m src.evaluate --skip-translation
python -m src.evaluate --skip-punctuation --skip-ordering \
    --max-translation-examples 50
python -m src.evaluate --max-punct-examples 300 --max-order-pairs 500 \
    --max-order-articles 10 --report /tmp/quick_eval.json
```

This is useful when iterating: punctuation is fast, ordering is slow, translation is optional.

### 22.6 Interpreting the results honestly

- The punctuation model is good at sentence endings, weak at commas, blind to question/exclamation.
- The ordering model is better than chance and better than TF-IDF, but full-order recovery is weak.
- The translation model is solid on a standard benchmark but varies by sentence.
- The pipeline outputs are usable, especially for prose; dialogue-like inputs are over-segmented.

> **Check yourself.** Why evaluate ordering with both pair accuracy and Kendall's tau? *Pair accuracy measures the classifier in isolation; tau measures the end-to-end ordering quality after decoding. A model can be decent at pairs but poor at full order, so both are needed.*

## 23. Reproducibility: from zero to outputs

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Provide two reproducible recipes: the full Colab/T4 run and the verified local subset run. Document exactly what was executed, save all reports, and publish an executed notebook. Never present a documented recipe as if it had been run.
</div>

### 23.1 Full Colab recipe

```bash
# 1. Environment
bash scripts/colab_setup.sh

# 2. Data
python -m src.data_prep --num-articles 3000 --max-train-pairs 40000

# 3. Train
python -m src.punctuation.train --epochs 3 --batch-size 8 --learning-rate 3e-4
python -m src.ordering.train    --epochs 2 --batch-size 8 --learning-rate 2e-4

# 4. Evaluate
python -m src.evaluate --max-translation-examples 200

# 5. Run
python -m src.pipeline --strategy borda
```

On a T4, this is the intended full run. The notebook automates the same sequence. The official IndicTrans2 checkpoint requires accepting the license and logging in.

### 23.2 Local verified recipe

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

This produced the adapters, the seven output sets, and the three reports used throughout this guide.

### 23.3 Chronology of the build

| Phase | What was done | Outcome |
| --- | --- | --- |
| 1 | Built the generic NLP lessons and environment check | classical tokenization/preprocessing foundation |
| 2 | Wrote a rule-based baseline (DP segmentation, TF-IDF ordering, OPUS-MT) | working but brittle; kept for comparison |
| 3 | Chose mBERT + LoRA for stages 1-2 and IndicTrans2 for stage 3 | fits the no-LLM and 1.5 GB constraints |
| 4 | Built the Wikipedia data pipeline | 3,000 articles; 25k windows; 40k pairs; held-out articles |
| 5 | Fixed dependency incompatibilities (transformers 5, peft/torch, datasets/hub) | pinned verified versions |
| 6 | Fixed MPS attention and IndicTrans2 tokenizer/config issues | local MPS run works |
| 7 | Trained subset adapters (4k/4k) | punctuation boundary F1 0.8633; ordering macro F1 0.6534 |
| 8 | Ran the full pipeline on all seven samples | 443 sentences; 1-2 minute stages on MPS |
| 9 | Evaluated all stages | boundary F1 0.8842; Borda tau 0.2007; BLEU 34.125 |
| 10 | Built and executed the demo notebook | visible outputs, all seven files aligned |
| 11 | Published the repository and notebook to GitHub | public repo |
| 12 | Attempted full-data local training | stopped; subset adapters remain; full recipe documented |

### 23.4 How to verify the results

```bash
# Compile and import
python -m compileall -q src
python -c 'import src.pipeline, src.evaluate, src.punctuation.train, src.ordering.train'

# Check training summaries
python - <<'PY'
import json
for path in ['models/punctuation/training_summary.json',
             'models/ordering/training_summary.json']:
    data = json.load(open(path))
    print(path, data['trainable_parameters'], data['total_parameters'])
    print('  test:', {k: v for k, v in data['test_metrics'].items()
                      if 'f1' in k or 'accuracy' in k})
PY

# Re-run individual evaluations
python -m src.evaluate --skip-ordering --skip-translation --report /tmp/punct.json
python -m src.evaluate --skip-punctuation --skip-translation --report /tmp/order.json
python -m src.evaluate --skip-punctuation --skip-ordering --report /tmp/trans.json

# End-to-end smoke test on one file
python -m src.pipeline --limit 1 --output-dir /tmp/pipeline-smoke
```

### 23.5 What is not done

- Full-data training was not completed locally. The adapters are subset adapters.
- The notebook has not been run top to bottom on a Colab T4 in this repository. The local demo notebook has been executed.
- There is no automated test suite (no pytest). Verification is via scripts and reports.
- The local IndicTrans2 mirror should be replaced with the official gated checkpoint for a final submission if access is available.

> **Check yourself.** Why is “it runs on my machine” not reproducibility? *Because dependencies, hardware kernels, seeds, and data versions can all differ. Reproducibility requires pinned versions, documented commands, saved reports, and a clear statement of what was and was not run.*

<div class='pagebreak'></div>

# Part III — The notebooks, cell by cell

## 24. `colab_end_to_end.ipynb` — the full beginner course

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Put the entire full-scale workflow in one notebook: install, prepare data, train both adapters, evaluate, run the pipeline, and save artifacts. The notebook is the executable version of the README recipe, but it also teaches the concepts behind each step instead of only listing commands.
</div>

This notebook has 59 cells: short explanations and examples interleaved with code. It is designed for a Colab T4 GPU and assumes no prior NLP experience. The first sections explain the problem, the later sections train and evaluate the models, and the final sections run the assignment pipeline and check the artifacts.

### 24.1 Section map

| Cells | Section | Purpose |
| ---: | --- | --- |
| 0‑2 | Big picture and prerequisites | What the project does, what the three stages are, and what the reader needs before starting. |
| 3‑5 | 0. Project location | Clone the public repo if needed, `os.chdir` into it, and add it to `sys.path`. |
| 6‑8 | 0.1 GPU check | Print torch/CUDA information and stop with a clear message if no GPU is present. |
| 9‑12 | 1. Install packages | Run `scripts/colab_setup.sh`, then import the installed libraries to prove the environment works. |
| 13‑14 | 1.1 Hugging Face access | Read an optional `HF_TOKEN` secret for the gated IndicTrans2 translation stage. |
| 15‑16 | 2. Settings and seed | Import project constants, call `set_seed(42)`, and cap GPU memory with `limit_gpu_memory(1.5)`. |
| 17‑26 | 3. Data | Explain inputs, targets and labels; show a raw assignment file; build the datasets; print the summary; run the data-only self-check. |
| 27‑34 | 4. Punctuation | Show subword tokenization, explain mBERT + LoRA and class weights, train stage 1, print the training summary, and list common fixes. |
| 35‑38 | 5. Ordering | Explain pairwise classification and the Borda/clusters/beam decoders, train stage 2, and print the training summary. |
| 39‑44 | 6. Evaluation | Explain held-out metrics, run `src.evaluate`, print the report, and run the pre-pipeline self-check. |
| 45‑49 | 7. Pipeline | Run the three stages on the seven assignment files, inspect groups, and explain what good output looks like. |
| 50‑51 | 8. Memory and timing | Read `run_summary.json` and explain the cost of each stage. |
| 52‑53 | 9. Final self-check | Run every verifier check against the full run. |
| 54‑55 | 10. Save results | Zip `outputs/` and copy `reports/` to a downloadable location. |
| 56‑58 | 11. Troubleshooting, next reading, glossary | Diagnose common failures, list the code-reading order, and define the vocabulary used in the notebook. |

### 24.2 Cell-by-cell notes

**Cells 0-2: orientation.** The notebook starts with a plain-English description of the problem: Hindi text has lost its punctuation and sentence order, and the pipeline must restore both before translating to English. It also states the prerequisites: a Google account, a T4 GPU runtime, and an optional Hugging Face token for gated IndicTrans2. There is no hidden setup step; if the GPU check or the install check fails, the notebook stops before doing expensive work.

**Cells 3-5: project folder.** The first code cell looks for an existing project root, falls back to `/content/nlp-zero-to-hero`, and runs `git clone` if the repository is missing. If the folder already exists, it pulls the latest `main` when possible, so a rerun does not keep stale code. It then changes into the root and inserts it into `sys.path`. This is why a reviewer can open the GitHub Colab link and use **Run all** without uploading a folder first.

**Cells 6-8: GPU diagnostics.** The notebook imports torch, prints the torch version, checks `torch.cuda.is_available()`, and prints the GPU name and total memory. The next markdown cell explains that a CPU runtime is not a valid substitute: training would take hours instead of minutes. This is a hard stop, not a warning.

**Cells 9-12: dependencies.** Cell 10 runs `bash scripts/colab_setup.sh`, which upgrades pip, installs `requirements-colab.txt`, and installs `IndicTransToolkit` from a wheel or a source build. Cell 12 imports `datasets`, `transformers`, `peft`, `torch` and `sacrebleu`, then prints their versions and the GPU name. The pins are deliberate: transformers 4.44.2 keeps the ONNX conversion path that IndicTrans2 needs, peft 0.13.2 avoids a macOS-only tensor import, and datasets 3.6.0 matches the pinned hub client.

**Cells 15-16: settings and memory.** The notebook imports `LABELS`, `PUNCT_DATA`, `ORDER_DATA`, `PUNCT_MODEL_DIR`, `ORDER_MODEL_DIR`, `TRANSLATION_MODEL`, `set_seed` and `limit_gpu_memory`. It calls `set_seed(42)` so splits and training are repeatable, then applies a 1.5 GB cap on a 15 GB T4 (a fraction of 0.1). The cap is intentionally tight; it turns a silent memory overrun into a clear CUDA out-of-memory error.

**Cells 17-26: data.** The markdown cells explain the difference between an input and a target, why Wikipedia can supply both punctuation labels and sentence order, and why Hindi needs care (the danda `।`, `str.split()` instead of a naive `\w` regex, and `NULL` removal). Cell 19 reads `data/raw/sample-1.txt` and prints its token count. Cells 20-21 define the five labels and describe the windowing/pair-building recipe. Cell 22 runs:

```bash
python -m src.data_prep --num-articles 3000 --max-train-pairs 40000
```

Cell 24 reads `data/processed/data_prep_summary.json` and prints article counts, split sizes, punctuation example counts, label distributions, ordering pair counts, and one held-out window. Cell 26 runs the data-only verifier:

```bash
python scripts/verify_colab_run.py --skip-models --skip-reports --skip-outputs
```

The expected counts are 3,000 articles; 2,700/150/150 splits; 25,000/2,317/2,208 punctuation windows; and 40,000/3,000/3,000 ordering pairs.

**Cells 27-34: punctuation.** Cell 29 uses `AutoTokenizer.from_pretrained("bert-base-multilingual-cased")` on a short Hindi phrase to show that a word becomes several subword tokens. The label is attached to the first subword; the remaining subword positions are ignored by the loss. The markdown explains why punctuation is imbalanced (most words are `none`), how class weights compensate, and what LoRA freezes and trains. Cell 31 runs:

```bash
python -m src.punctuation.train --epochs 3 --batch-size 8 \
    --learning-rate 3e-4 --memory-gb 1.5
```

Cell 33 reads `models/punctuation/training_summary.json` and prints trainable parameters, validation metrics, test metrics, and peak memory. Cell 34 lists the most common failure modes: missing labels, a missing adapter, and CUDA OOM.

**Cells 35-38: ordering.** The markdown explains why predicting a whole permutation is impractical (20 sentences give more than 2 quintillion orders) and why the project instead trains a pairwise classifier on `(A, B) -> 1` and `(B, A) -> 0`. It compares the three decoders: Borda (default, robust), clusters (TF-IDF + KMeans), and beam (higher cost, sometimes better). Cell 36 runs:

```bash
python -m src.ordering.train --epochs 2 --batch-size 8 \
    --learning-rate 2e-4 --memory-gb 1.5
```

Cell 38 prints validation/test metrics from `models/ordering/training_summary.json`. Pair accuracy should be clearly above 0.5; the full-data recipe should beat the subset result.

**Cells 39-44: evaluation.** The markdown distinguishes word accuracy from boundary F1 (why the latter is the meaningful punctuation score), pair accuracy from Kendall's tau (why a good pair classifier can still produce a mediocre full order), and BLEU/chrF from accuracy (they are comparative, not percentages). Cell 40 runs `python -m src.evaluate` with the notebook's caps. Cell 42 prints `reports/evaluation.json`. Cell 44 runs the verifier again, this time skipping outputs and the adapter smoke test:

```bash
python scripts/verify_colab_run.py --skip-outputs --skip-model-smoke
```

If IndicTrans2 is not authenticated, translation fails with a clear error; the notebook does not silently fall back to another model.

**Cells 45-49: pipeline.** Cell 46 runs:

```bash
python -m src.pipeline --strategy borda --memory-gb 1.5
```

The pipeline removes `NULL`, restores punctuation, groups sentences, orders each group, translates five sentences at a time, and writes four files per input. Cell 48 prints the first Hindi and English groups of sample-1. Each group should contain five sentences, except possibly the last. Cell 49 explains what to look for: Hindi punctuation present, English readable, and occasional ordering mistakes expected because ordering is the weakest stage.

**Cells 50-51: memory and timing.** Cell 51 reads `reports/run_summary.json` and prints the strategy, timings, memory limit, and per-file counts. On a T4, ordering is the slowest stage; on the local MPS run, ordering took 130.53 seconds, translation 102.17 seconds, and punctuation 2.31 seconds.

**Cells 52-53: final self-check.** Cell 53 runs the full verifier with no skip flags. It checks the data, model files, reports, deterministic adapter smoke test, metric thresholds, and every output file. The notebook does not declare success unless this cell finishes with `All checks passed.`

**Cells 54-55: saving.** Cell 55 zips `outputs/` to `/content/hindi_assignment_outputs.zip` and copies `reports/` to `/content/hindi_assignment_reports/`. These are the artifacts to attach to a submission.

**Cells 56-58: troubleshooting and vocabulary.** The final cells cover GPU problems, 401/gated-repo errors, toolkit build failures, import errors after install, CUDA OOM, slow data downloads, and failed self-checks. The code-reading order is `src/constants.py`, `src/pipeline.py`, the two model packages, the translator wrapper, then `src/data_prep.py` and `src/evaluate.py`. The glossary defines adapter, batch, BLEU, Borda, chrF, epoch, F1, inference, Kendall's tau, LoRA, loss, mBERT, NULL, seed, and softmax.

### 24.3 How to run it

1. Open the notebook in Colab.
2. `Runtime > Change runtime type > Hardware accelerator: GPU (T4)`.
3. Press **Runtime > Run all**. The first code cell clones the repository to `/content/nlp-zero-to-hero` automatically.
4. Optional, for translation: accept the IndicTrans2 licence on Hugging Face and add a Colab secret named `HF_TOKEN`. Training works without it; evaluation and the pipeline need it.
5. The notebook installs dependencies, prepares data, trains both adapters, evaluates, runs the pipeline and saves the artifacts.
6. Download the zip and reports from `/content/`.

> **Common issue.** The notebook's default `TRANSLATION_MODEL` is the official gated Hub id. If you have not authenticated, translation fails at the evaluation or pipeline step with a clear error. Add `HF_TOKEN` in Colab Secrets and rerun the optional login cell. The executed demo notebook is the tested evidence path on this machine.

## 25. `pipeline_demo.ipynb` — the executed local demo

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Publish an executed notebook that uses the real adapters and shows visible outputs for every stage. This is the evidence artifact: it proves the local pipeline runs, that Hindi/English outputs align, and that the reported metrics are reproducible on this machine.
</div>

This notebook has 22 cells and is already executed. It is the short, beginner-friendly companion to the full course: it does not train anything, but it explains each stage and shows real predictions and real output files.

### 25.1 Cell map and observed outputs

| Cell | Type | What it does | Observed output |
| ---: | --- | --- | --- |
| 0 | markdown | Title, what the project does, the three stages, and environment summary | Python 3.12, torch 2.15 MPS, 2-3 minutes |
| 1 | markdown | Section 0: how the notebook finds the project root | — |
| 2 | code | Set the project root, add it to `sys.path`, print the Python version | `Python: 3.12.13` |
| 3 | markdown | What the path cell should print and why it matters | — |
| 4 | markdown | Section 1: punctuation problem, five labels, subword labels, LoRA | — |
| 5 | code | Load the punctuation adapter and predict one test window | 73 words; token accuracy 0.849; correct `है -> sentence_end`, `राज्यव्यवस्था -> comma` |
| 6 | markdown | How to read `true=` and `pred=`; why the classifier warning is expected | — |
| 7 | markdown | Section 2: why pairwise ordering, and how Borda turns pairs into an order | — |
| 8 | code | Load the ordering adapter and score one test pair | gold 1; P(A before B)=0.711; P(B before A)=0.289 |
| 9 | markdown | How to read the pair probabilities; why one pair is only a sanity check | — |
| 10 | markdown | Section 3: IndicTrans2, language tags, and why there is no fallback | — |
| 11 | code | Translate three short Hindi sentences | “India is a huge country.”; “It's a test.”; “Please submit your report by tomorrow.” |
| 12 | markdown | What the translations should look like; wording may vary | — |
| 13 | markdown | Section 4: the four output files per sample and the alignment rule | — |
| 14 | code | Check alignment of all seven samples | 52/11, 55/11, 49/10, 53/11, 55/11, 80/16, 99/20; all aligned |
| 15 | markdown | How to read `aligned=True`; what good and bad output look like | — |
| 16 | markdown | Section 5: what each metric means and why the subset scores are honest | — |
| 17 | code | Print the key metrics from `reports/evaluation.json` | 0.8724 / 0.8842; 0.6543; 0.2007; 34.125 / 59.686 |
| 18 | markdown | How to read the scores; why ordering is the weak point | — |
| 19 | markdown | Section 6: reproduce the artifacts and run the verifier | — |
| 20 | markdown | The three command-line commands | — |
| 21 | markdown | Section 7: code-reading order and exercises | — |

### 25.2 Why the demo matters

- It uses the **actual trained adapters**, not toy weights.
- It shows **visible outputs** for all three stages, including Hindi text.
- It verifies **alignment**: the number of Hindi sentences equals the number of English sentences equals the summary's sentence count, and the same for groups.
- It reports the **subset adapter** scores honestly instead of hiding the weak ordering result.
- It is short enough for a reviewer to read before the full course. In Colab it is **read-only evidence**: the model weights and processed data are git-ignored, so a reviewer should read the saved outputs instead of using **Run all**.

### 25.3 How to run it

```bash
source .venv.backup-20261109-124948/bin/activate
jupyter notebook notebooks/pipeline_demo.ipynb
```

Select the `nlp-zero-to-hero-mps` kernel. Run all cells. It takes about 2-3 minutes because it loads both adapters and the local IndicTrans2 model.

> **Note on the warning messages.** Loading the base model with a new classification head prints “Some weights ... were not initialized”. This is expected; the adapter supplies the trained head. The warning does not mean the model is broken.

## 26. The lesson notebooks and legacy baseline

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Keep the learning artifacts and the rejected baseline in the repository, clearly labeled. They show the path from classical NLP to the final architecture and make the comparison honest, but they are not imported by the final pipeline.
</div>

### 26.1 Lessons

The three lesson notebooks are generic English NLP exercises:

- **Lesson 01:** imports NumPy and pandas and prints versions. It is an environment smoke test.
- **Lesson 02:** uses NLTK `word_tokenize` and `sent_tokenize` on an English sentence. It demonstrates that tokenization is not trivial: punctuation is separated, contractions may split, sentences are detected.
- **Lesson 03:** uses `lessons/preprocessing.py` to clean text (lowercase, remove URLs, mentions, hashtags, punctuation), tokenize, remove stopwords, stem with Porter, and lemmatize with WordNet. It is a classical preprocessing pipeline, not used by the final Hindi models.

`lessons/train.py` is a TF-IDF + logistic regression text classifier. It expects `data/raw/dataset.csv` with `text` and `label` columns, which is not part of the assignment data. It is an educational example.

### 26.2 Baseline

The baseline is a complete rule-based pipeline:

- **Segmentation:** dynamic programming over candidate boundaries, scored by verb endings, sentence starters, `NULL` hints, and a length prior.
- **Ordering:** TF-IDF char 2-4 grams + KMeans + greedy medoid ordering.
- **Translation:** `Helsinki-NLP/opus-mt-hi-en` MarianMT.

It works, but it is brittle: the segmentation rules are hand-tuned, the ordering is lexical, and the translation model is generic. The final pipeline replaces every heuristic with a trained model. The baseline remains useful as a comparison and as a record of the engineering path.

> **Check yourself.** Why keep the baseline if it is not used? *Because it documents the alternatives and provides a sanity check. If the trained pipeline were worse than a simple rule-based system, that would be important to know. It also shows the evolution of the solution, which is valuable in a project report.*
