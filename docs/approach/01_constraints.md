# 2. The assignment and the constraints

The design is a response to a specific problem, not a general architecture looking for a use. This section states the problem precisely, then lists every constraint that shaped the solution.

## 2.1 The task

Given a Hindi text file whose words are space-separated:

- punctuation has been removed,
- sentence order has been shuffled,
- the placeholder token `NULL` may appear,
- no sentence boundaries are marked,

produce:

1. the same Hindi words with punctuation and sentence boundaries restored,
2. the sentences in a plausible original order,
3. an English translation of each ordered sentence,
4. the English and Hindi sentences grouped in blocks of five.

The output contract is four files per input:

```text
sample-N.sentences_hi.txt   one Hindi sentence per line
sample-N.groups_hi.txt      the same sentences, five per line
sample-N.sentences_en.txt   one English sentence per line
sample-N.groups_en.txt      the same translations, five per line
```

Line $i$ in `sentences_hi.txt` must correspond to line $i$ in `sentences_en.txt`. The group count must be $\lceil n/5 \rceil$ for $n$ sentences.

## 2.2 Hard constraints

| Constraint | Consequence for the design |
| --- | --- |
| **No LLM** | The project may not call a hosted chat model or use a generative LLM to punctuate, order, or translate. Every stage must be a task-specific model. |
| **1.5 GB GPU memory** | The pipeline may not keep all three models resident. It must stage, quantize, or otherwise bound peak memory. |
| **Hindi input, English output** | Tokenization, labels, and translation must handle Devanagari and the danda `।`. |
| **No word rewriting** | Punctuation must not change the input words. Translation is the only stage allowed to produce new words. |
| **Exact alignment** | Sentence and group files must line up; no heuristic re-alignment after translation. |
| **Reproducibility** | Splits, seeds, dependencies, commands, and reports must be explicit. |

The no-LLM and memory constraints are the two that most strongly determine the architecture. A single large model is the obvious alternative, and it fails both.

## 2.3 Soft constraints and engineering goals

| Goal | Why it matters | How the approach handles it |
| --- | --- | --- |
| Per-stage evaluation | A single end-to-end score hides which stage failed. | Boundary F1, pair F1/tau, BLEU/chrF. |
| Failure isolation | If ordering is weak, punctuation and translation should still run. | Separate models and separate stages. |
| Small artifacts | Adapters should be easy to save, ship, and reload. | LoRA adapters and saved tokenizers. |
| Determinism where possible | Reports should be reproducible. | Seed 42, article-level splits, fixed decoding. |
| Honest reporting | Subset results must not be presented as full-data results. | Explicit subset/full claim boundary. |
| Local execution | No data should leave the machine. | Local mBERT and IndicTrans2 mirror. |

## 2.4 What “fit 1.5 GB” actually means

The cap is not a slogan. On CUDA the code calls:

```python
torch.cuda.set_per_process_memory_fraction(1.5 / total_gpu_memory_gb)
```

which makes the CUDA allocator raise an out-of-memory error instead of exceeding the fraction. On CPU and MPS this call is a no-op because the assignment budget is defined for the CUDA runtime.

The architecture reduces memory in five ways:

1. **LoRA** — the 178M-parameter backbone is frozen; only about 0.33% of parameters are trained.
2. **Gradient checkpointing** — activations are recomputed instead of stored.
3. **fp16 on CUDA** — the model and activations use half precision.
4. **One model at a time** — punctuation, ordering, and translation never coexist.
5. **Bounded batch sizes** — 8 for training, 16 for punctuation inference, 32 for pair scoring, 4 for translation.

The local MPS run reports about 0.67 GB allocated for each adapter stage. That is evidence about the adapters, **not** evidence that the CUDA cap was exercised. The document distinguishes those claims.

## 2.5 Why the constraints point to decomposition

A single model that reads the raw token stream and emits punctuated, ordered English would need to learn three different structures at once. It would also need aligned training data that does not exist here: Wikipedia gives punctuation and order, but not a parallel triplet of (unpunctuated shuffled Hindi, punctuated ordered Hindi, English).

Decomposition solves four problems simultaneously:

| Problem | How decomposition solves it |
| --- | --- |
| Supervision | Each stage trains on a signal that naturally exists: punctuation labels, consecutive sentences, and Hindi-English pairs. |
| Evaluation | Each stage has its own held-out metric and baseline. |
| Memory | Only one model is resident at a time. |
| Failure isolation | A weak orderer does not prevent punctuation from being measured or translation from running. |

This is the thesis of the whole project. Section 3 expands it.

<div class="pagebreak"></div>

# 3. Design thesis: decompose, do not monolith

<div class="decision">

**APPROACH CHOSEN.** Decompose the assignment into three specialised stages instead of asking one model to do everything. The decomposition is not a convenience; it is the only design that satisfies the no-LLM, 1.5 GB, supervision, evaluation, and alignment constraints at the same time.

</div>

## 3.1 The four reasons for decomposition

| Reason | Explanation |
| --- | --- |
| **Constraint fit** | A single generative model large enough to do all three tasks well would not fit 1.5 GB and would resemble an LLM, which the assignment forbids. |
| **Supervision availability** | Wikipedia gives punctuation labels and natural sentence order for free. It does not give aligned triplets of unpunctuated shuffled Hindi, punctuated ordered Hindi, and English. |
| **Evaluation** | Decomposition yields one metric per stage: boundary F1, pair F1/Kendall's tau, and BLEU/chrF. A single score cannot localise a failure. |
| **Failure isolation** | If ordering is weak, punctuation and translation remain valid and measurable. If translation is unavailable, the first two stages still produce a partial result. |

<figure class="diagram">
<img src="approach_assets/01_thesis.svg" alt="Monolithic versus decomposed design">
<figcaption><strong>Figure 3.1.</strong> The monolithic alternative is rejected because it has no aligned training data, opaque errors, and a budget risk. The decomposed pipeline gives each stage its own model, metric, and artifact.</figcaption>
</figure>

## 3.2 What is shared and what is separate

The decomposition is not a set of unrelated scripts. Three things are shared:

- **One article-level split** — both learned stages use the same train/validation/test documents, so no article leaks across stages and the evaluation is comparable.
- **One backbone family** — mBERT is used for token classification and pair classification. Only the head and task change.
- **One orchestration contract** — every stage consumes and produces plain Python lists, so the pipeline can load, free, and replace models independently.

Three things are deliberately separate:

- **Training objectives** — token-level cross entropy for punctuation, pair-level cross entropy for ordering.
- **Adapters** — two independent LoRA adapters, saved separately.
- **Decoding** — per-word argmax for punctuation, global ranking for ordering, beam search for translation.

## 3.3 The design invariants

These invariants hold at every point in the pipeline:

1. **The word stream is preserved after `NULL` removal.** Stage 1 cannot insert, delete, or rewrite a word.
2. **A sentence boundary is a label, not a character edit.** The punctuation mark is reconstructed after prediction.
3. **Ordering permutes sentences, not words.** The permutation is applied to the punctuated and plain sentence lists consistently.
4. **Translation preserves line order.** A flat batch is translated and then sliced back by per-file sentence counts.
5. **Grouping is pure chunking.** No sorting or filtering happens after translation.
6. **One model is resident per stage.** The previous model is closed before the next is loaded.

If a change violates one of these invariants, it changes the approach rather than improving it.

## 3.4 What decomposition costs

Decomposition is not free:

- **Error propagation.** A punctuation boundary error changes the input to ordering; an ordering error changes the context seen by translation.
- **No joint optimisation.** Each stage optimises its own objective, not the final output.
- **More artifacts.** Two adapters, one translator, one tokenizer family, and one preprocessing toolkit must all be present.
- **More latency.** Models are loaded and freed sequentially; the ordering stage can be quadratic in the number of sentences.

These costs are real and are analysed in Section 12 and Section 14. The design accepts them because the constraints make the monolith worse.

<div class="pagebreak"></div>

# 4. System overview and contracts

This section describes the pipeline as a system: what enters, what leaves, how control flows, and where each responsibility lives.

## 4.1 End-to-end control flow

<figure class="diagram">
<img src="approach_assets/02_pipeline.svg" alt="End-to-end pipeline">
<figcaption><strong>Figure 4.1.</strong> Every input file is read once. The three stages run in sequence, and the outputs are written as sentence files and groups of five.</figcaption>
</figure>

The pipeline executes in this order:

1. Read every `data/raw/*.txt` file once.
2. Remove exact `NULL` tokens and record how many were removed.
3. Load the punctuation adapter, predict labels for every file, rebuild punctuated and plain sentences, then free the model.
4. Load the ordering adapter, score and order the plain sentences, apply the same permutation to the punctuated sentences, then free the model.
5. Load IndicTrans2, translate all ordered Hindi sentences as one flat list, then free the model.
6. Slice translations back by each file's sentence count.
7. Write four files per input and a run summary.

The single read is deliberate: all three stages must see exactly the same token order, and re-reading files between stages would create an opportunity for drift.

## 4.2 Input contract

| Property | Requirement | Why |
| --- | --- | --- |
| Encoding | UTF-8 | Devanagari support |
| Token separator | whitespace | matches the assignment format |
| Placeholder | exact token `NULL` | removed before stage 1 |
| Punctuation | none expected | stage 1 predicts it |
| Sentence order | shuffled | stage 2 recovers it |
| File extension | `.txt` | the pipeline globs `*.txt` |

The raw reader is intentionally simple:

```python
text = Path(path).read_text(encoding="utf-8")
raw_tokens = text.split()
words = [token for token in raw_tokens if token != "NULL"]
```

It does not strip punctuation or normalise Unicode. That is correct for this assignment because the input is already word-tokenised and unpunctuated.

## 4.3 Output contract

For each input stem, four files are written:

| File | Contents | Alignment rule |
| --- | --- | --- |
| `sentences_hi.txt` | ordered Hindi sentences, one per line | line $i$ is sentence $i$ |
| `groups_hi.txt` | the same sentences, five per line | group $g$ contains sentences $5g+1$ to $5g+5$ |
| `sentences_en.txt` | English translations, one per line | line $i$ translates Hindi line $i$ |
| `groups_en.txt` | the same translations, five per line | group $g$ contains translations $5g+1$ to $5g+5$ |

The group count is always $\lceil n/5 \rceil$. The last group may be shorter.

## 4.4 The record data structure

The pipeline keeps one dictionary per input:

```text
record = {
  "words": [...],                # after NULL removal
  "null_count": int,
  "labels": [...],               # stage 1 label ids
  "sentences_hi": [...],         # punctuated Hindi sentences
  "sentences_plain": [...],      # same sentences without punctuation
  "ordered_hi": [...],           # stage 2 permutation applied
  "groups_hi": [[...], ...],     # groups of five
  "ordered_en": [...],           # stage 3 output
  "groups_en": [[...], ...],
}
```

The key design detail is the distinction between `sentences_hi` and `sentences_plain`. Ordering is trained on plain sentences, so it must score the plain versions. The resulting permutation is then applied to the punctuated versions, so the Hindi output retains its punctuation.

## 4.5 Failure model

The pipeline uses a **fail-loud** policy:

| Failure | Behaviour |
| --- | --- |
| No punctuation adapter | `FileNotFoundError` with the expected path and training command |
| No ordering adapter | `FileNotFoundError` with the expected path and training command |
| Missing `IndicTransToolkit` | `ImportError` with the install command |
| Gated IndicTrans2 repo | upstream authentication error; no fallback |
| No input files | `SystemExit` with the resolved input directory |
| CUDA OOM | allocator error; no silent CPU fallback |
| Empty translation item | preserved as an empty output line so alignment survives |

There is deliberately **no fallback translator**. A fallback would hide a failure and make metrics incomparable across runs.

## 4.6 Implementation map

| Responsibility | File | Key functions / classes |
| --- | --- | --- |
| Configuration | `src/constants.py` | paths, labels, window sizes, model ids |
| Dataset construction | `src/data_prep.py` | `parse_article`, `make_punctuation_windows`, `split_article_ids` |
| Pair construction | `src/ordering/pairs.py` | `make_pairs_from_sentences` |
| Punctuation data | `src/punctuation/dataset.py` | `PunctuationDataset` |
| Punctuation training | `src/punctuation/train.py` | `WeightedTrainer`, `class_weights` |
| Punctuation inference | `src/punctuation/predict.py` | `PunctuationRestorer` |
| Ordering data | `src/ordering/pairs.py` | `SentencePairDataset` |
| Ordering training | `src/ordering/train.py` | `Trainer` with pair metrics |
| Ordering inference | `src/ordering/decode.py` | `OrderScorer`, `order_from_matrix` |
| Translation | `src/translation/indictrans.py` | `IndicTrans2Translator` |
| Orchestration | `src/pipeline.py` | `run`, `read_tokens` |
| Evaluation | `src/evaluate.py` | punctuation, pair, tau, BLEU/chrF |
| Verification | `scripts/verify_colab_run.py` | data, model, report, output checks |

<div class="pagebreak"></div>
