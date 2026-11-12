# 9. System architecture: one path, three isolated stages

<div class="decision">

**APPROACH CHOSEN.** Run three stages in a fixed order — punctuation, ordering, translation — with an explicit text contract between them. Load one model at a time, release it before loading the next, and write every result through one orchestration path. No stage has a fallback model, and no stage may silently change the sentence count.

</div>

## 9.1 The end-to-end contract

<figure class="diagram">
<img src="approach_assets/02_pipeline.svg" alt="End-to-end pipeline">
<figcaption><strong>Figure 9.1.</strong> Tokens enter once. Stage 1 restores punctuation and boundaries; stage 2 permutes whole sentences; stage 3 translates each sentence; the writer emits sentence and group files. Models are loaded and released sequentially.</figcaption>
</figure>

The pipeline is deliberately linear:

```text
raw token list
  -> remove NULL
  -> punctuation labels + Hindi sentences + plain sentences
  -> permutation of whole sentences
  -> ordered Hindi sentences
  -> one English translation per Hindi sentence
  -> sentence files + groups of five
```

Each arrow is a data contract, not just a function call.

| Hand-off | Producer | Consumer | Guarantee |
| --- | --- | --- | --- |
| token list | `read_tokens` | punctuation restorer | every non-`NULL` token appears exactly once |
| labels + sentences | `PunctuationRestorer.restore` | ordering stage | labels align with words; sentences cover all words |
| plain sentences | same | `order_sentences` | no punctuation in ordering input |
| permutation | `order_sentences` | writer / translator | whole sentences only; no word-level movement |
| Hindi sentence list | ordering stage | translator | one Hindi string per English string |
| English list | translator | writer | same length and order as Hindi list |
| groups | `group_lines` | writer | consecutive chunks of exactly five, last may be shorter |

The most important guarantee is **index alignment**. Ordering returns indices into the punctuated sentence list. Translation returns one string per input string. Grouping is a pure slice. At no point does a stage guess which output belongs to which input.

## 9.2 Stage isolation and memory

The 1.5GB budget is enforced by not holding models concurrently:

```text
load punctuation model -> process all files -> delete model + empty cache
load ordering model    -> process all files -> delete model + empty cache
load translation model -> process all files -> delete model + empty cache
```

<figure class="diagram">
<img src="approach_assets/08_memory.svg" alt="Sequential model loading and memory budget">
<figcaption><strong>Figure 9.2.</strong> The stages do not overlap. Each model gets the full budget; peak memory is the largest single stage, not the sum of all stages.</figcaption>
</figure>

The source of truth for the cap is `src/constants.py`:

```python
MEMORY_LIMIT_GB = 1.5
```

`limit_gpu_memory` turns that into a CUDA process fraction:

```python
fraction = min(1.0, 1.5 / total_gb)
torch.cuda.set_per_process_memory_fraction(fraction)
```

On MPS and CPU the function returns a no-op report. This is an honest limitation: the 1.5GB cap is enforced on CUDA, not measured locally on Metal.

`free_memory()` calls `gc.collect()` and, on CUDA, `torch.cuda.empty_cache()` and `torch.cuda.reset_peak_memory_stats()`. The pipeline also deletes the stage object before loading the next model. Python imports for later stages happen inside `run`, so importing the pipeline does not load all three model families at once.

### 9.2.1 Measured local memory

| Stage | Device | Allocated at end |
| --- | --- | ---: |
| punctuation training | MPS | 0.6687 GB |
| ordering training | MPS | 0.6709 GB |
| translation training | — | not trained |
| translation inference | MPS | not recorded in `run_summary.json` |

The training summaries record adapter-training memory, not pipeline peak memory. The pipeline report records time and file counts, not memory. Do not claim a measured end-to-end peak.

## 9.3 Stage interfaces

### 9.3.1 Stage 1: punctuation

Input: `list[str]` of words with `NULL` already removed.

Output:

```python
labels:          list[str]   # one of none/comma/sentence_end/question/exclamation
sentences_hi:    list[str]   # punctuated sentences
sentences_plain: list[str]   # same sentences without punctuation
```

The plain list exists because the ordering model was trained on plain text. The punctuated list is what the final output and translation consume.

### 9.3.2 Stage 2: ordering

Input: plain sentences.

Output: a permutation `list[int]`.

The pipeline applies the permutation to `sentences_hi`:

```python
ordered_hi = [sentences_hi[index] for index in order]
```

This is why ordering cannot lose punctuation: it never edits strings, only reorders references.

### 9.3.3 Stage 3: translation

Input: ordered Hindi sentences from all files, flattened into one list.

Output: one English string per input, in the same order.

The translator preserves empty inputs as empty outputs. That detail matters because an empty sentence must not shift the index of the next one.

### 9.3.4 Writer

For each input file, four outputs are written:

```text
<stem>.sentences_hi.txt
<stem>.groups_hi.txt
<stem>.sentences_en.txt
<stem>.groups_en.txt
```

`groups_hi.txt` joins each consecutive group of five Hindi sentences with spaces. `groups_en.txt` does the same for English. The last group may be shorter than five.

<figure class="diagram">
<img src="approach_assets/11_groups.svg" alt="Sentence and translation alignment contract">
<figcaption><strong>Figure 9.3.</strong> Sentence $i$ in maps to sentence $i$ out. Grouping is a later, purely positional chunking step; no sorting or filtering happens after translation.</figcaption>
</figure>

## 9.4 Configuration

All paths are absolute, derived from the repository root, so the code behaves the same from a notebook, a CLI, or Colab.

| Constant | Value | Purpose |
| --- | --- | --- |
| `BASE_MODEL` | `bert-base-multilingual-cased` | shared encoder for stages 1 and 2 |
| `TRANSLATION_MODEL` | `ai4bharat/indictrans2-indic-en-dist-200M` | stage 3 |
| `SRC_LANG` / `TGT_LANG` | `hin_Deva` / `eng_Latn` | translation tags |
| `WIKI_DATASET` | `wikimedia/wikipedia` | data source |
| `WIKI_CONFIG` | `20231101.hi` | pinned snapshot |
| `WINDOW_WORDS` | 64 | punctuation window |
| `STRIDE_WORDS` | 40 | punctuation overlap |
| `MIN_WORDS` / `MAX_WORDS` | 32 / 96 | window bounds |
| `MIN_SENT_WORDS` / `MAX_SENT_WORDS` | 5 / 40 | sentence filter |
| `MAX_SEQ_LENGTH` | 128 | punctuation tokenizer bound |
| `ORDER_MAX_LENGTH` | 192 | pair tokenizer bound |
| `GROUP_SIZE` | 5 | output grouping |
| `MEMORY_LIMIT_GB` | 1.5 | CUDA process cap |
| `SEED` | 42 | reproducibility |

The dataclass-free constants file is intentionally simple: one place to audit every magic number.

## 9.5 CLI and reproduction sequence

```bash
python -m src.data_prep \
  --num-articles 3000 \
  --max-train-windows 25000 \
  --max-train-pairs 40000 \
  --eval-articles 40

python -m src.punctuation.train \
  --epochs 3 --batch-size 8 --learning-rate 3e-4

python -m src.ordering.train \
  --epochs 2 --batch-size 8 --learning-rate 2e-4

python -m src.evaluate

python -m src.pipeline --strategy borda
```

Useful variants:

| Command | Effect |
| --- | --- |
| `python -m src.pipeline --limit 1` | process one raw file |
| `python -m src.pipeline --no-translate` | run stages 1–2 only |
| `python -m src.pipeline --strategy beam` | use beam decoding |
| `python -m src.pipeline --strategy clusters` | use TF-IDF clusters + beam |
| `python -m src.evaluate --skip-translation` | fast metric check |
| `python -m src.punctuation.train --max-train-samples 4000` | subset adapter |
| `python -m src.ordering.train --max-train-samples 4000` | subset adapter |

## 9.6 Artifacts and folder structure

```text
data/raw/                    assignment .txt files
data/processed/
  data_prep_summary.json     counts, label shares, split sizes
  punctuation/
    train.jsonl              windows with words + labels
    val.jsonl
    test.jsonl
  ordering/
    train.jsonl              sentence pairs with 0/1 labels
    val.jsonl
    test.jsonl
    eval_articles.jsonl      complete articles for Kendall's tau
models/
  punctuation/
    adapter/                 LoRA weights, classifier, tokenizer
    training_summary.json
  ordering/
    adapter/
    training_summary.json
  translation/
    indictrans2-indic-en-200M/
outputs/
  sample-N.sentences_hi.txt
  sample-N.groups_hi.txt
  sample-N.sentences_en.txt
  sample-N.groups_en.txt
reports/
  data_prep_summary.json
  evaluation.json
  run_summary.json
  punctuation_training.json
  ordering_training.json
```

The split between `models/` and `reports/` matters: adapters are runtime artifacts; reports are evidence. Training writes both a model-local summary and a central report copy.

## 9.7 Failure behaviour

| Failure | Behaviour |
| --- | --- |
| Missing punctuation adapter | `FileNotFoundError` with a training command |
| Missing ordering adapter | `FileNotFoundError` with a training command |
| Missing IndicTransToolkit | `ImportError` with setup instructions |
| Empty input file | punctuation emits no sentences; downstream stages see empty lists |
| Empty sentence string | preserved as empty translation; indices do not shift |
| Unknown decoding strategy | `ValueError` listing valid strategies |
| No `.txt` files | `SystemExit` with the resolved input directory |
| One sentence | ordering returns identity; translation still runs |
| Gated upstream model | requires the local mirror |

There are no silent fallbacks. A missing model fails loudly instead of producing a different pipeline.

## 9.8 Reproducibility controls

- `SEED = 42` seeds Python, NumPy, PyTorch, and CUDA.
- Article shuffle, window shuffle, pair sampling, and evaluation shuffle all use seeded RNGs.
- Data snapshot is pinned to `20231101.hi`.
- Training uses fixed class weights derived from the data, not random resampling.
- Decoders are deterministic given the score matrix.
- Beam search and clustering have fixed random states.
- The evaluation report records model paths, dataset id, example counts, and metrics.
- The run report records group size, strategy, model paths, memory limit, timings, and per-file counts.

## 9.9 Pros of the system design

1. **Auditable.** Every hand-off has a named contract.
2. **Memory-safe.** One model at a time fits the 1.5GB target.
3. **No hidden fallback.** Failures are visible.
4. **Alignment-safe.** Sentence indices are preserved end to end.
5. **Reproducible.** One seed and one pinned dataset snapshot.
6. **Modular.** Each stage can be evaluated independently.
7. **Notebook-friendly.** The same classes drive the CLI and the demos.

## 9.10 Cons and system risks

| Risk | Cause | Mitigation |
| --- | --- | --- |
| **No cross-stage feedback** | Linear pipeline; later stages cannot fix earlier mistakes. | Evaluate each stage; document compounding error. |
| **Ordering dominates runtime** | Quadratic pair scoring. | Borda is the cheapest decoder; grouping is not used to reduce pairs. |
| **Memory cap unverified locally** | MPS no-op. | CUDA fraction code is implemented; be explicit that it was not exercised here. |
| **Translation has no fallback** | Single model path. | Intentional: auditable failure over silent substitution. |
| **Sentence-level translation loses context** | Contract is one sentence in, one out. | Document limitation; future paragraph mode. |
| **Pipeline peak memory not recorded** | `run_summary` stores timings only. | Add peak memory logging before claiming a measured peak. |
| **Subset adapters** | Shipped models are small. | Separate subset results from full-recipe claims. |

<div class="pagebreak"></div>

# 10. Training system and evaluation feedback loop

<div class="decision">

**APPROACH CHOSEN.** Train two independent LoRA adapters on the same frozen mBERT backbone, select each by a task-appropriate validation metric, and evaluate with one report that separates punctuation, ordering, and translation. The report is the feedback loop: it decides which decoder ships and which weaknesses are documented rather than hidden.

</div>

## 10.1 Two adapters, one backbone

| Property | Punctuation adapter | Ordering adapter |
| --- | --- | --- |
| Task | 5-class token classification | 2-class pair classification |
| Backbone | mBERT | mBERT |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.1 | 0.1 |
| Target modules | query, value | query, value |
| Saved modules | classifier | classifier |
| Trainable params | 593,669 | 591,362 |
| Total params | 177,860,362 | 178,446,340 |
| Selection metric | validation boundary F1 | validation accuracy |
| Default epochs | 3 | 2 |
| Default batch | 8 | 8 |
| Default LR | 3e-4 | 2e-4 |

The adapters are separate. Sharing the backbone weights on disk would save space but does not change the fact that the tasks need different heads and different adaptation.

## 10.2 Why LoRA

| Option | Pros | Cons |
| --- | --- | --- |
| **LoRA** | Tiny checkpoints; small trainable set; fits memory; fast to train | Slightly lower ceiling than full fine-tuning |
| Full fine-tuning | Potentially stronger | Much more memory; larger checkpoints; wasteful for a frozen 178M encoder |
| Frozen features + linear head | Cheapest | Too weak for punctuation and pair classification |
| Prompt tuning | Very small | Less mature for BERT classification |

LoRA is the right trade-off for a 1.5GB budget and a project that must ship adapters, not multi-gigabyte checkpoints.

## 10.3 Training data flow

```text
Wikipedia snapshot
  -> parse articles
  -> article-level split
  -> punctuation windows (words + labels)
  -> ordering pairs (A, B, label)
  -> adapters
  -> evaluation report
  -> decoder choice
```

The feedback loop is explicit:

1. Train both adapters with default or subset caps.
2. Run `python -m src.evaluate`.
3. Read punctuation boundary F1, ordering pair accuracy, decoder tau, and BLEU/chrF.
4. Ship Borda because it wins the decoder comparison.
5. Document weak classes and weak decoders instead of retuning until they look good.

## 10.4 Evaluation report design

`reports/evaluation.json` has three top-level sections:

```json
{
  "no_llm": true,
  "memory_limit_gb": 1.5,
  "punctuation": {"status": "ok", "accuracy": 0.8724, "boundary": {}, "per_class_f1": {}},
  "ordering": {"status": "ok", "pairs": {}, "kendall_tau": {}},
  "translation": {"status": "ok", "bleu": 34.125, "chrf": 59.686, "samples": []}
}
```

Why one report?

- one command reproduces the scoreboard,
- every metric has a recorded dataset and example count,
- skipped sections are marked, not silently omitted,
- the pipeline claim and the metric claim stay tied together.

`reports/run_summary.json` is the runtime counterpart: it records timings, file counts, group size, strategy, model paths, and memory limit. Training summaries live under `models/*/training_summary.json` and are copied to `reports/*_training.json`.

## 10.5 What the training loop does not do

- It does not search hyperparameters automatically.
- It does not use an LLM for data generation or evaluation.
- It does not train stage 3; IndicTrans2 is used as supplied.
- It does not train on the assignment files.
- It does not claim that subset adapters are the full recipe.
- It does not hide negative results such as zero question/exclamation F1.

That restraint is part of the approach: the system is small enough to explain end to end, and every number has a known provenance.

## 10.6 Training risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| MPS SDPA dropout bug | Training fails or silently misbehaves | eager attention on MPS |
| fp16 on MPS | Unsupported/incorrect | fp32 on MPS; fp16 CUDA |
| Rare classes | Zero F1 | class weights, honest reporting |
| Subset caps | Lower scores | separate subset/full claims |
| Validation metric mismatch | Wrong checkpoint selected | boundary F1 for punctuation, accuracy for ordering |
| Dataset drift | Different label counts | pin snapshot, record summary |
| Long windows/pairs | Truncation | bounds at 128/192 subwords |

<div class="pagebreak"></div>
