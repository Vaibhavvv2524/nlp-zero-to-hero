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

## 24. `colab_end_to_end.ipynb` — the full recipe

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Put the entire full-scale workflow in one notebook: install, prepare data, train both adapters, evaluate, run the pipeline, and save artifacts. The notebook is the executable version of the README recipe, not a separate demo.
</div>

This notebook has 28 cells: markdown explanations interleaved with code. It is designed for a Colab T4 GPU.

### 24.1 Cell map

| Cell | Type | Purpose |
| ---: | --- | --- |
| 0 | markdown | title, stage table, GPU instruction |
| 1 | markdown | project location instructions |
| 2 | code | `os.chdir` to project, `sys.path` setup |
| 3 | code | torch/CUDA diagnostics |
| 4 | markdown | dependency install explanation |
| 5 | code | `!bash scripts/colab_setup.sh` |
| 6 | markdown | imports, seeds, memory-cap explanation |
| 7 | code | imports, `set_seed`, `limit_gpu_memory(1.5)` |
| 8 | markdown | data-preparation explanation |
| 9 | code | `!python -m src.data_prep --num-articles 3000 --max-train-pairs 40000` |
| 10 | code | read and print data summary; show a sample |
| 11 | markdown | stage 1 explanation (labels, LoRA, class weights) |
| 12 | code | `!python -m src.punctuation.train --epochs 3 --batch-size 8 ...` |
| 13 | code | read punctuation training summary |
| 14 | markdown | stage 2 explanation (pair format, decoders) |
| 15 | code | `!python -m src.ordering.train --epochs 2 --batch-size 8 ...` |
| 16 | code | read ordering training summary |
| 17 | markdown | evaluation explanation |
| 18 | code | `!python -m src.evaluate --max-...` |
| 19 | code | print evaluation JSON |
| 20 | markdown | pipeline explanation (four files per input, groups of five) |
| 21 | code | `!python -m src.pipeline --strategy borda --memory-gb 1.5` |
| 22 | code | print the first four Hindi/English groups of sample-1 |
| 23 | markdown | memory/timing explanation |
| 24 | code | read and print `run_summary.json` |
| 25 | markdown | save-results explanation |
| 26 | code | zip outputs, copy reports |
| 27 | markdown | notes, limitations, troubleshooting |

### 24.2 Cell-by-cell notes

**Cells 0-3: setup and diagnostics.** The notebook sets `PROJECT_DIR = '/content/nlp-zero-to-hero'`, checks that it exists, changes into it, and inserts it into `sys.path`. It then prints the torch version, whether CUDA is available, the GPU name, and total GPU memory. If CUDA is not available, the training cells will be extremely slow; switch the runtime to GPU.

**Cell 5: dependencies.** It runs `scripts/colab_setup.sh`, which upgrades pip, installs `requirements-colab.txt`, and installs `IndicTransToolkit` (wheel or source). It also prints torch and CUDA information. This cell takes a few minutes on a fresh runtime.

**Cell 7: seeds and memory cap.** It imports the project constants and utilities, calls `set_seed(42)`, and applies `limit_gpu_memory(1.5)`. On a 15 GB T4, this sets the process memory fraction to `1.5 / 15 = 0.1`. If the process tries to exceed it, CUDA raises an out-of-memory error rather than silently using more.

**Cells 8-10: data.** Cell 9 builds the datasets. Cell 10 reads `data/processed/data_prep_summary.json`, prints article counts, split sizes, punctuation examples and label counts, ordering pair counts, and shows the first 18 words/labels of a training window. The expected counts are 3,000 articles, 2,700/150/150 splits, 25,000/2,317/2,208 windows, and 40,000/3,000/3,000 pairs.

**Cells 11-13: punctuation training.** Cell 12 trains for 3 epochs with batch size 8 and LR 3e-4. Cell 13 reads `models/punctuation/training_summary.json` and prints trainable parameters, validation metrics, test metrics, and peak memory. The log also prints the label counts and class weights.

**Cells 14-16: ordering training.** Cell 15 trains for 2 epochs with batch size 8 and LR 2e-4. Cell 16 prints validation/test metrics and peak memory. The expected pair accuracy is above 0.5; the full-data recipe should improve on the subset result.

**Cells 17-19: evaluation.** Cell 18 runs `src.evaluate` with caps: 1,200 punctuation examples, 3,000 ordering pairs, 30 articles, 200 translation examples. Cell 19 prints the evaluation JSON. If IndicTrans2 is not authenticated, the translation stage is skipped with a reason; the punctuation and ordering stages still run.

**Cells 20-22: pipeline.** Cell 21 runs `src.pipeline` with Borda and the 1.5 GB cap. Cell 22 prints the first four Hindi and English groups of sample-1. Each group should contain five sentences, except possibly the last.

**Cells 23-24: timing and memory.** Cell 24 reads `run_summary.json` and prints the strategy, timings, memory limit, and per-file counts. On a T4, the ordering stage is the slowest; on MPS, the local run took 130.53 seconds for ordering, 102.17 seconds for translation, and 2.31 seconds for punctuation.

**Cells 25-26: saving.** Cell 26 zips `outputs/` to `/content/hindi_assignment_outputs.zip` and copies `reports/` to `/content/hindi_assignment_reports/`. This is what you attach to a submission.

**Cell 27: notes.** It restates the no-LLM guarantee, the no-fallback guarantee, the memory strategy, the data limitations, the rarity of question/exclamation marks, the difficulty of ordering, and translation evaluation caveats. It also has troubleshooting bullets for toolkit builds, peft/transformers mismatches, and CUDA OOM.

### 24.3 How to run it

1. Open the notebook in Colab.
2. `Runtime > Change runtime type > Hardware accelerator: GPU (T4)`.
3. Upload the project folder to `/content/nlp-zero-to-hero` or clone the GitHub repo there.
4. If using the official IndicTrans2, accept the license on Hugging Face and run `huggingface-cli login` (or set `HF_TOKEN`) before cell 21/18.
5. `Runtime > Run all`.
6. Download the zip and reports from `/content/`.

> **Common issue.** The notebook's default `TRANSLATION_MODEL` is the official gated Hub id. If you have not authenticated, translation fails. You can pass a local mirror path in the commands, but the notebook cells as written use the constant. The local demo notebook is the tested path on this machine.

## 25. `pipeline_demo.ipynb` — the executed local demo

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Publish an executed notebook that uses the real adapters and shows visible outputs for every stage. This is the evidence artifact: it proves the local pipeline runs, that Hindi/English outputs align, and that the reported metrics are reproducible on this machine.
</div>

This notebook has 13 cells and is already executed. It is the artifact that proves the local pipeline runs with visible outputs.

### 25.1 Cell map and observed outputs

| Cell | Type | What it does | Observed output |
| ---: | --- | --- | --- |
| 0 | markdown | title, environment summary | Python 3.12, torch 2.15 MPS, 2-3 minutes |
| 1 | code | set project root, print Python version | `Python: 3.12.13` |
| 2 | markdown | stage 1 explanation | — |
| 3 | code | load punctuation adapter, predict one test window | 73 words; token accuracy 0.849; correct `है -> sentence_end`, `राज्यव्यवस्था -> comma` |
| 4 | markdown | stage 2 explanation | — |
| 5 | code | load ordering adapter, score one test pair | gold 1; P(A before B)=0.711; P(B before A)=0.289 |
| 6 | markdown | stage 3 explanation | — |
| 7 | code | translate three sentences | “India is a huge country.”; “It's a test.”; “Please submit your report by tomorrow.” |
| 8 | markdown | end-to-end artifacts explanation | — |
| 9 | code | check alignment of all seven samples | 52/11, 55/11, 49/10, 53/11, 55/11, 80/16, 99/20; all aligned |
| 10 | markdown | evaluation explanation | subset adapters |
| 11 | code | print key metrics | 0.8724 / 0.8842; 0.6543; 0.2007; 34.125 / 59.686 |
| 12 | markdown | reproduce commands | — |

### 25.2 Why the demo matters

- It uses the **actual trained adapters**, not toy weights.
- It shows **visible outputs** for all three stages, including Hindi text.
- It verifies **alignment**: the number of Hindi sentences equals the number of English sentences equals the summary's sentence count, and the same for groups.
- It reports the **subset adapter** scores honestly.

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
