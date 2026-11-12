# Part IV — Results, defenses, and reference

<div class='pagebreak'></div>

## 27. Results and limitations

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Report every stage separately, state exactly what was trained (subset vs full), and explain what the numbers do and do not mean. Never hide a weak metric behind an aggregate.
</div>

### 27.1 Headline table

| Stage | Metric | Value | What it means |
| --- | --- | --- | --- |
| Punctuation | token accuracy | 0.8724 | most words, which are unpunctuated, are classified correctly |
| Punctuation | boundary F1 | 0.8842 | precision/recall on `sentence_end`, `question`, `exclamation` |
| Punctuation | comma F1 | 0.316 | commas are hard; many are missed |
| Punctuation | question/excl F1 | 0.000 | too rare in training (236 and 153 examples) |
| Ordering | pair accuracy | 0.6543 | better than chance, better than TF-IDF |
| Ordering | macro F1 | 0.6534 | balanced across both pair orientations |
| Ordering | Borda Kendall tau | 0.2007 | weak-to-moderate full-order recovery |
| Ordering | clusters tau | 0.1223 | cluster decoding underperforms |
| Ordering | beam tau | 0.0921 | beam variant underperforms with this model |
| Ordering | TF-IDF tau | 0.0631 | lexical baseline is much weaker |
| Translation | BLEU | 34.125 | acceptable general-purpose output on 100 IITB sentences |
| Translation | chrF | 59.686 | character-level overlap is reasonably high |

### 27.2 Interpretation

**Punctuation is mostly a sentence-end problem.** The model is strong at deciding where a sentence ends. It is weak at commas, because comma placement is stylistic and context-dependent, and the training data has only 3.3% commas. Question and exclamation marks are effectively absent, so no model can learn them from this data. The high token accuracy is partly an artifact of the `none` majority: predicting `none` everywhere would already score about 0.91. Boundary F1 is the honest metric.

**Ordering is better than lexical but not solved.** Pair accuracy of 0.65 means the model has learned real coherence signals. But full-order recovery via Borda is weak (tau 0.20). Pairwise preferences are noisy and can conflict, and a wrong early decision propagates. Borda is the best of the three decoders; that is why it is the default.

**Translation is the strongest stage.** BLEU 34 on IITB is a reasonable result for a 200M-parameter Indic→English model on general prose. It is not a claim about this specific noisy, reordered input. Translation is also the least affected by the upstream stages on well-formed sentences, because IndicTrans2 is robust to imperfect punctuation.

### 27.3 Limitations, stated plainly

1. **Subset adapters.** The shipped adapters were trained on 4,000 punctuation windows and 4,000 ordering pairs, not the full 25,000/40,000. The full recipe is documented, but full-data training was not completed locally.
2. **Synthetic supervision.** Punctuation labels and ordering pairs come from Wikipedia text. The model learns Wikipedia style. Dialogue, poetry, and informal text behave differently.
3. **Rare labels.** Question and exclamation marks are almost absent. Do not expect them.
4. **Ordering ceiling.** With a 64-word window and pairwise BERT scoring, full-document order remains hard. The grouping step hides some errors by keeping groups of five in the same relative order, but it does not fix them.
5. **Translation evaluation.** BLEU/chrF are computed on clean IITB sentence pairs, not on the pipeline's reordered, punctuated output. They measure the translator, not the full pipeline.
6. **Gated model.** The official IndicTrans2 requires license acceptance. A local mirror was used. Results may differ slightly from the official checkpoint.
7. **No test suite.** Verification is by scripts, reports, and the executed notebook, not by unit tests.
8. **Single seed.** All results are from one run with `SEED=42`. Variance is not measured.

### 27.4 What would improve it most

| Change | Expected effect | Cost |
| --- | --- | --- |
| Full-data training (25k/40k) | clear improvement in both adapters | GPU hours |
| Class-balanced sampling for punctuation | better comma recall | retraining |
| Add question/exclamation data (or augment) | non-zero rare-class F1 | data work |
| Better ordering decoder (e.g., insertion sort with beam) | higher tau than Borda | inference only |
| Train ordering on longer contexts | better document-level order | memory/time |
| Evaluate translation on pipeline output with references | true end-to-end quality | annotation effort |
| Unit tests for alignment and windowing | fewer regressions | engineering time |

> **Check yourself.** Why is “boundary F1 0.88” a better headline than “accuracy 0.87”? *Because 91% of words have no punctuation, so accuracy is dominated by the majority class. Boundary F1 measures the decisions that actually matter.*

## 28. Design decisions, expanded

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Make each major decision explicit, with the alternative, the reason, the pros, and the cons. This is the material a reviewer or examiner will probe, so it is written as a decision log rather than prose.
</div>

### D1. Language: Hindi

- **Chosen:** Hindi (Devanagari), with English translation.
- **Alternatives:** a higher-resource language with better tooling.
- **Why:** the task requires Hindi; Devanagari has clear sentence-ending danda, rich morphology, and available multilingual models.
- **Pros:** direct task match; the danda gives an unambiguous sentence terminator; multilingual models support Devanagari; only one target language is needed.
- **Cons:** fewer high-quality tools than English; rich morphology and combining marks make tokenization harder; no large Hindi-only pretrained encoder was available locally.

### D2. Tokenization: whitespace, not regex

- **Chosen:** `str.split()`.
- **Alternatives:** `re.findall(r'\w+', text)`.
- **Why:** Devanagari vowel signs are combining marks; `\w+` drops them (`अयोध्या` → `अय`, `ध`, `य`). Whitespace keeps words intact.
- **Pros:** preserves Devanagari vowel signs; trivially simple; matches the assignment's space-separated input; no regex or language-specific rules.
- **Cons:** punctuation attached to tokens must be stripped or modelled; no Unicode normalization; clitics and compounds are not split; whitespace errors pass through.

### D3. Punctuation: token classification, not seq2seq

- **Chosen:** classify each word into five labels.
- **Alternatives:** generate punctuated text with a seq2seq model; use a large LLM.
- **Why:** one-to-one alignment, fast, small, no hallucination, fits 1.5 GB.
- **Pros:** exact one-to-one word alignment; fast; small; deterministic; cannot hallucinate or delete words; easy to evaluate per token.
- **Cons:** cannot insert or rewrite words; rare labels are hard to learn; comma style is limited; subword-to-word alignment adds complexity.

### D4. Model: mBERT + LoRA

- **Chosen:** `bert-base-multilingual-cased` with LoRA r=16 on query/value.
- **Alternatives:** full fine-tuning; a larger multilingual model; XLM-R; MuRIL.
- **Why:** mBERT has Hindi coverage, is ~178M params, and LoRA makes training cheap. It was available locally.
- **Pros:** Hindi coverage; only ~178M parameters; LoRA makes training cheap; adapters are portable and small; one backbone serves two tasks.
- **Cons:** older and weaker than XLM-R/MuRIL; the cased vocabulary fragments Devanagari into many subwords; 512-token limit; no generative ability.

### D5. Label alignment: first subword

- **Chosen:** assign the word label to its first subword; ignore the rest.
- **Alternatives:** average subword logits; predict on the last subword; use word-level tokenizer.
- **Why:** Bengali/Hindi word endings carry punctuation cues, but first-subword is simple, stable, and works well in practice.
- **Pros:** simple and stable; one label per word; works directly with WordPiece; no extra parameters or alignment model.
- **Cons:** ignores evidence in later subwords; no learned pooling; sensitive to tokenizer changes; a word's final subword often carries the clearest punctuation cue.

### D6. Ordering: pairwise binary classification + Borda

- **Chosen:** train `[CLS] A [SEP] B [SEP] → before/after`; decode with Borda.
- **Alternatives:** pointer networks; seq2seq ordering; listwise ranking; topological sort.
- **Why:** pairwise data is easy to generate and the classifier is small and robust. Borda aggregates noisy preferences.
- **Pros:** training data is easy to generate; the model generalizes to any document length; small and robust; decoders can be swapped without retraining; performance is measurable with pair accuracy.
- **Cons:** O(n²) pairs; preferences can be cyclic; Borda is an approximation, not the optimal Kemeny order; full-order accuracy is weak; early errors propagate.

### D7. Ordering training data: balanced both directions

- **Chosen:** for each adjacent pair, add `(A,B,1)` and `(B,A,0)`.
- **Alternatives:** only natural order; sample random pairs.
- **Why:** removes position bias and doubles data.
- **Pros:** removes position bias; doubles the training data; forces the model to learn symmetric coherence rather than a direction shortcut.
- **Cons:** artificial reverse pairs can be plausible; the model cannot use absolute position; training and inference cost double.

### D8. Translation: IndicTrans2 200M

- **Chosen:** `ai4bharat/indictrans2-indic-en-dist-200M`.
- **Alternatives:** MarianMT; NLLB; a large LLM; Google Translate API.
- **Why:** Indic-specialized, 200M is feasible, strong Hindi→English quality, runs locally.
- **Pros:** Indic-specialized; 200M parameters fit the budget; strong Hindi→English quality; runs locally; exact sentence-level alignment.
- **Cons:** gated license; ~913 MB; 256-token context limit; slow on CPU; no document-level context; a local mirror may differ from the official checkpoint.

### D9. Memory: staged load/free vs simultaneous

- **Chosen:** load one model, run its stage on all files, free it, load the next.
- **Alternatives:** load all three and stream.
- **Why:** three models would exceed 1.5 GB; staging caps peak memory at the largest model.
- **Pros:** peak memory is the largest single model, not the sum; simple and predictable; easy to swap or skip a stage; no quantization needed.
- **Cons:** repeated load/free time; more I/O; cannot jointly optimize stages; failures are only discovered when the stage runs.

### D10. Evaluation: per-stage, honest subsets

- **Chosen:** held-out windows/pairs/articles and a 100-sentence translation sample.
- **Alternatives:** one end-to-end human evaluation.
- **Why:** fast, reproducible, isolates failures.
- **Pros:** fast; reproducible; isolates which stage failed; cheap to rerun; uses the same inference code as the pipeline.
- **Cons:** does not measure full-pipeline quality directly; translation is scored on clean IITB pairs, not pipeline output; subset sizes add sampling variance.

### D11. Notebooks: recipe + executed demo

- **Chosen:** a full Colab recipe notebook and an executed local demo.
- **Alternatives:** only scripts; only a notebook.
- **Why:** the recipe is portable; the demo is evidence.
- **Pros:** the recipe is portable to Colab; the demo is executed evidence; readers can inspect outputs without training; markdown explains each step.
- **Cons:** two artifacts to maintain; the full recipe is not executed in this repository; versions can drift; Colab-specific paths and installs.

### D12. No LLM

- **Chosen:** no generative LLM anywhere in the pipeline.
- **Alternatives:** prompt GPT/Gemini for punctuation/ordering/translation.
- **Why:** the assignment forbids an LLM-based solution; also cost, latency, privacy, and determinism.
- **Pros:** satisfies the assignment constraint; lower cost and latency; privacy-preserving; deterministic; small dependency surface; every prediction is traceable to a model.
- **Cons:** lower quality ceiling, especially for ordering; no fluent generation; more engineering than prompting; weak rare-punctuation handling.

## 29. Viva and interview question bank

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Prepare short, defensible answers that connect each design choice to a constraint and a measurement. If you can answer the “why not” questions, you understand the project.
</div>

### 29.1 Problem and data

1. **What is the task?** Restore punctuation in unpunctuated Hindi, reorder shuffled sentences into a coherent document, and translate to English.
2. **Why is Hindi hard?** Rich morphology, free word order, and combining marks in Devanagari.
3. **Where does the data come from?** 3,000 Hindi Wikipedia articles, split 2,700/150/150.
4. **How are punctuation labels derived?** From the original punctuation attached to tokens; `none/comma/sentence_end/question/exclamation`.
5. **How are windows built?** 64-word windows, stride 40, kept only if 32-96 words and 5-40-word sentences.
6. **Why stride 40?** It gives overlap so boundary words appear in multiple windows and get averaged.
7. **How are ordering pairs built?** Adjacent sentence pairs, both directions, balanced labels.
8. **Why balanced directions?** To remove position bias and double the data.
9. **Why hold out articles for ordering?** Pairs from the same article would leak context into evaluation.
10. **What is the `NULL` token?** A placeholder in the raw data; it is removed before processing and counted.

### 29.2 Punctuation

11. **Why token classification?** It preserves alignment and is fast, small, and deterministic.
12. **Why not seq2seq?** It can hallucinate or change words, and alignment is harder.
13. **How do you handle subwords?** Use the first subword's prediction for the word.
14. **Why first subword?** Simple and stable; a learned pooling is future work.
15. **Why class weights?** `none` is ~91%; weights `[0.22, 6.13, 3.47, 20, 20]` counter imbalance.
16. **Why cap weights at 20?** Uncapped `question` would be ~1540, destabilizing training.
17. **How does sliding-window inference work?** Predict per window, accumulate logits per word, average, argmax.
18. **Why `window_weight`?** It counts how many windows scored each word so averaging is correct.
19. **What are boundary labels?** `sentence_end`, `question`, `exclamation`; `none` and `comma` are not boundaries.
20. **Why is comma F1 low?** Commas are stylistic, sparse, and context-dependent.
21. **Why are question/exclamation F1 zero?** Only 236/153 training examples; the model never predicts them.
22. **What is `restore_punctuation`?** A lightweight function that attaches punctuation for display, capped at one terminal mark.
23. **Why not predict punctuation for every word independently in the pipeline?** That is exactly what it does; the labels come from averaged window logits.
24. **What would improve commas?** More comma data, balanced sampling, or a dedicated comma model.
25. **How is punctuation evaluated?** 1,500 test windows, token accuracy, boundary P/R/F1, per-class F1.

### 29.3 Ordering

26. **Why pairwise?** Easy supervision, small model, and it generalizes to any document length.
27. **What is the input format?** `[CLS] sentence A [SEP] sentence B [SEP]`.
28. **What is the label?** 1 if A precedes B, 0 otherwise.
29. **Why `ORDER_MAX_LENGTH=192`?** Two 90-token sentences plus separators fit.
30. **What is Borda count?** Each pair vote adds to the winner's score; sort by score.
31. **Why Borda over topological sort?** Preferences can be cyclic; Borda tolerates cycles.
32. **What are the three decoders?** Borda, clusters (greedy), beam (incremental).
33. **Which won?** Borda: tau 0.2007 vs clusters 0.1223 and beam 0.0921.
34. **Why is tau low?** Pairwise noise, cycles, and error propagation.
35. **What is Kendall's tau?** Rank correlation between predicted and true order.
36. **Why a TF-IDF ablation?** It isolates how much the neural model adds over lexical overlap.
37. **What did the ablation show?** TF-IDF tau 0.0631, much lower than the model.
38. **How is ordering evaluated?** Pair accuracy/macro F1 on 3,000 pairs, plus tau on 40 shuffled articles.
39. **Why 40 articles?** It is enough for a stable tau estimate and cheap to run.
40. **What is the complexity?** O(n²) pairs per article; for 100 sentences, ~4,950 pairs.

### 29.4 Translation

41. **Why IndicTrans2?** Indic-specialized, strong Hindi→English, 200M is feasible.
42. **What is the language code pair?** `hin_Deva` → `eng_Latn`.
43. **Why the toolkit?** It handles Indic tokenization/spacing and the model's processor.
44. **Why not pass `src_lang`/`tgt_lang` to the tokenizer?** This model does not take them; passing them breaks.
45. **What is `_import_indic_processor`?** A compatibility shim for transformers 4 vs 5 import paths.
46. **Why `return_attention_mask`?** IndicTrans2's forward needs it explicitly.
47. **Why `clean_up_tokenization_spaces=False`?** It preserves Indic spacing.
48. **Why a local mirror?** The official checkpoint is gated; a mirror was available.
49. **What are BLEU and chrF?** n-gram precision/recall and character n-gram F-score.
50. **Why evaluate on IITB?** It is a standard Hindi-English benchmark with references.
51. **What is the 256-token limit?** The model's max positions; long sentences are truncated.

### 29.5 Engineering

52. **Why whitespace tokenization?** `\w+` drops Devanagari matras.
53. **Why pin transformers 4.44.2?** transformers 5 removes the ONNX/tokenizer base path IndicTrans2 needs.
54. **Why pin peft 0.13.2?** Newer peft imports `torch.distributed.tensor`, absent on macOS torch.
55. **Why pin datasets 3.6.0?** datasets 4/5 require huggingface-hub ≥ 1.31, conflicting with the stack.
56. **Why eager attention on MPS?** SDPA with dropout is broken on MPS.
57. **Why fp32 on MPS?** fp16 is CUDA-only in this stack.
58. **How is memory capped?** `torch.cuda.set_per_process_memory_fraction(1.5/total)`.
59. **Why stage the models?** Loading all three exceeds 1.5 GB.
60. **Why no fallback model?** Silent fallbacks hide broken setups and hurt reproducibility.
61. **Why no LLM?** Assignment constraint; also cost, latency, determinism.
62. **How is the pipeline aligned?** Same sentence boundaries and `GROUP_SIZE` for Hindi and English.
63. **Why `GROUP_SIZE=5`?** Groups are readable and match the expected output format.
64. **What happens to `NULL`?** Removed, counted, reported.
65. **How do you reproduce?** Pinned env, documented commands, saved reports, executed demo.

<div class='pagebreak'></div>

# Appendices

## Appendix A. Command reference

```bash
# Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-colab.txt
pip install IndicTransToolkit==1.1.1

# Data
python -m src.data_prep --num-articles 3000 --max-train-pairs 40000

# Training
python -m src.punctuation.train --epochs 2 --batch-size 16 \
    --learning-rate 3e-4 --max-train-samples 4000
python -m src.ordering.train --epochs 1 --batch-size 16 \
    --learning-rate 2e-4 --max-train-samples 4000

# Evaluation
python -m src.evaluate \
    --translation-model models/translation/indictrans2-indic-en-200M \
    --max-translation-examples 100

# Pipeline
python -m src.pipeline \
    --translation-model models/translation/indictrans2-indic-en-200M \
    --translation-batch-size 8 --strategy borda

# Focused evaluations
python -m src.evaluate --skip-ordering --skip-translation --report /tmp/p.json
python -m src.evaluate --skip-punctuation --skip-translation --report /tmp/o.json
python -m src.evaluate --skip-punctuation --skip-ordering --report /tmp/t.json

# Smoke test
python -m src.pipeline --limit 1 --output-dir /tmp/smoke

# Build the guide
python scripts/build_study_guide.py
```

## Appendix B. Hyperparameters

| Parameter | Value |
| --- | --- |
| `BASE_MODEL` | `bert-base-multilingual-cased` |
| `TRANSLATION_MODEL` | `ai4bharat/indictrans2-indic-en-dist-200M` |
| `SRC_LANG` / `TGT_LANG` | `hin_Deva` / `eng_Latn` |
| `MAX_SEQ_LENGTH` | 128 |
| `ORDER_MAX_LENGTH` | 192 |
| `WINDOW_WORDS` / `STRIDE_WORDS` | 64 / 40 |
| `MIN_WORDS` / `MAX_WORDS` | 32 / 96 |
| `MIN_SENT_WORDS` | 5 |
| `MAX_SENT_WORDS` | 40 |
| `GROUP_SIZE` | 5 |
| `SEED` | 42 |
| LoRA r / alpha / dropout | 16 / 32 / 0.1 |
| LoRA targets | `query`, `value` |
| Punctuation LR / batch / epochs | 3e-4 / 16 / 2 (subset) |
| Ordering LR / batch / epochs | 2e-4 / 16 / 1 (subset) |
| Memory cap | 1.5 GB |

## Appendix C. Dataset statistics

| Split | Articles | Notes |
| --- | --- | --- |
| train | 2,700 | window/pair generation |
| val | 150 | model selection |
| test | 150 | evaluation |

| Dataset | Train | Val | Test |
| --- | ---: | ---: | ---: |
| Punctuation windows | 25,000 | 2,317 | 2,208 |
| Ordering pairs | 40,000 | 3,000 | 3,000 |
| Ordering eval articles | — | — | 40 |

| Label | Train count | Share |
| --- | ---: | ---: |
| none | 1,653,107 | 90.9% |
| comma | 59,324 | 3.3% |
| sentence_end | 104,840 | 5.8% |
| question | 236 | ~0.01% |
| exclamation | 153 | ~0.01% |
| total words | 1,817,660 | 100% |

## Appendix D. Report schemas

`reports/data_prep_summary.json` (values from the verified local run):

```json
{
  "dataset": "wikimedia/wikipedia",
  "config": "20231101.hi",
  "seed": 42,
  "articles": 3000,
  "splits": {"train": 2700, "val": 150, "test": 150},
  "punctuation": {
    "train": {"articles": 2700, "examples": 25000,
              "labels": {"none": {"count": 1653107, "share": 0.90947},
                         "comma": {"count": 59324, "share": 0.03264},
                         "sentence_end": {"count": 104840, "share": 0.05768},
                         "question": {"count": 236, "share": 0.00013},
                         "exclamation": {"count": 153, "share": 0.00008}}},
    "val": {"articles": 150, "examples": 2317},
    "test": {"articles": 150, "examples": 2208}
  },
  "ordering": {
    "train": {"articles": 2700, "pairs": 40000},
    "val": {"articles": 150, "pairs": 3000},
    "test": {"articles": 150, "pairs": 3000}
  }
}
```

`models/punctuation/training_summary.json` (abridged; ordering has the analogous keys):

```json
{
  "base_model": "bert-base-multilingual-cased",
  "epochs": 2,
  "batch_size": 16,
  "learning_rate": 0.0003,
  "train_windows": 4000,
  "trainable_parameters": 593669,
  "total_parameters": 177860362,
  "validation_metrics": {"eval_loss": 0.4399,
                         "eval_boundary_f1": 0.8246},
  "test_metrics": {"test_accuracy": 0.8688,
                   "test_boundary_f1": 0.8633},
  "label_counts": {"none": 1653107, "comma": 59324,
                   "sentence_end": 104840,
                   "question": 236, "exclamation": 153},
  "class_weights": [0.22, 6.128, 3.467, 20.0, 20.0],
  "memory_limit": 1.5,
  "memory_at_end": 0.0
}
```

`reports/run_summary.json`:

```json
{
  "group_size": 5,
  "ordering_strategy": "borda",
  "punctuation_model": "models/punctuation",
  "ordering_model": "models/ordering",
  "translation_model": "models/translation/indictrans2-indic-en-200M",
  "memory_limit_gb": 1.5,
  "timings_seconds": {"punctuation": 2.31,
                      "ordering": 130.53,
                      "translation": 102.17},
  "files": {"sample-1": {"words": 747, "null_tokens_removed": 5,
                         "sentences": 52, "groups": 11}}
}
```

## Appendix E. Dependency error table

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ImportError: transformers.onnx` | transformers 5 removed it | pin 4.44.2 |
| `ModuleNotFoundError: torch.distributed.tensor` | peft too new for macOS torch | pin peft 0.13.2 |
| `huggingface-hub` conflict | datasets 4/5 requires hub ≥ 1.31 | pin datasets 3.6.0 |
| SDPA dropout error on MPS | MPS SDPA bug | use eager attention |
| fp16 error on MPS | fp16 is CUDA-only here | use fp32 |
| CUDA OOM on T4 | three models or large batches | stage models, cap memory, smaller batch |
| Gated model 401 | no license acceptance | accept license, log in, or use mirror |
| Toolkit build failure | Rust/compiler missing on Colab | use wheel or source build per setup script |
| Tokenizer `src_lang` error | IndicTrans2 does not accept it | call tokenizer without `src_lang`/`tgt_lang` |
| Text extraction shows broken matras | PDF font/ToUnicode issue | visual shaping is correct; avoid text search |

## Appendix F. References and further study

- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2019).
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021).
- Gala et al., *IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages* (2023).
- Vaswani et al., *Attention Is All You Need* (2017).
- Koehn, *Neural Machine Translation* (2020).
- Jurafsky and Martin, *Speech and Language Processing* (3rd ed. draft).
- Hugging Face `transformers`, `peft`, `datasets` documentation.
- AI4Bharat `IndicTrans2` and `IndicTransToolkit` repositories.
- `cfilt/iitb-english-hindi` dataset on Hugging Face.
- NLTK book, chapters on tokenization and stemming.

## Appendix G. Study plan

**Week 1 — foundations.** Tokens, embeddings, attention, transformer blocks, BERT pretraining. Run the three lesson notebooks. Be able to draw a transformer block from memory.

**Week 2 — finetuning and adapters.** Classification heads, cross-entropy, class weights, LoRA math, training loops. Read `src/punctuation/train.py` and `src/ordering/train.py` line by line.

**Week 3 — the data pipeline.** Wikipedia parsing, sentence splitting, sliding windows, label alignment, balanced pairs, splits. Re-run `src.data_prep` and inspect the JSONL.

**Week 4 — inference and decoding.** Sliding-window prediction, Borda/clusters/beam, Kendall tau. Re-run `src.evaluate` and reproduce the reported numbers.

**Week 5 — translation and orchestration.** IndicTrans2, tokenizer quirks, staged loading, memory caps. Run the pipeline end to end and inspect all four outputs per file.

**Week 6 — communication.** Read this guide's Parts 0.5 and IV, answer the viva bank aloud, and explain one design decision to someone else in two minutes.

<div class='pagebreak'></div>

## Appendix H. Abbreviations and full forms

Every acronym, initialism, and short form used in this guide is listed here. The goal is that a reader never has to guess what an abbreviation means. Clipped words such as “config”, “repo”, and “demo” are ordinary shorthand, not acronyms; document notation such as `A.1` and `D1` is explained in H.6.

### H.1 Core NLP and machine learning

| Short form | Full form | Meaning in this guide |
| --- | --- | --- |
| **NLP** | Natural Language Processing | the field concerned with making computers process human language |
| **LLM** | Large Language Model | a large generative model such as GPT; explicitly not used here |
| **BERT** | Bidirectional Encoder Representations from Transformers | the pretrained encoder family used for stages 1 and 2 |
| **mBERT** | Multilingual BERT | `bert-base-multilingual-cased`, pretrained on many languages including Hindi |
| **XLM-R** | Cross-lingual Language Model – RoBERTa | a stronger multilingual encoder considered as an alternative |
| **MuRIL** | Multilingual Representations for Indian Languages | an Indic-focused BERT variant considered as an alternative |
| **GPT** | Generative Pre-trained Transformer | the decoder-only model family behind many LLMs |
| **LoRA** | Low-Rank Adaptation | the parameter-efficient fine-tuning method used for both adapters |
| **PEFT** | Parameter-Efficient Fine-Tuning | the library and the general family of methods that includes LoRA |
| **MLM** | Masked Language Modeling | BERT pretraining task: predict randomly masked tokens |
| **NSP** | Next Sentence Prediction | BERT pretraining task: predict whether B follows A |
| **FFN** | Feed-Forward Network | the position-wise two-layer network inside a transformer block |
| **MLP** | Multi-Layer Perceptron | a stack of linear layers with nonlinearities; the FFN is an MLP |
| **GELU** | Gaussian Error Linear Unit | a smooth activation function used in BERT |
| **ReLU** | Rectified Linear Unit | the simple `max(0, x)` activation mentioned as a contrast |
| **BPE** | Byte Pair Encoding | a subword tokenization algorithm |
| **OOV** | Out-of-Vocabulary | a word the tokenizer cannot represent directly |
| **TF-IDF** | Term Frequency–Inverse Document Frequency | a classical lexical weighting scheme used as an ordering ablation |
| **DP** | Dynamic Programming | the optimization method used by the rule-based baseline segmenter |
| **NP-hard** | Nondeterministic Polynomial-time hard | a complexity class; finding the optimal Kemeny order is NP-hard |
| **seq2seq** | Sequence-to-Sequence | a model that maps one sequence to another; used for translation |
| **SDPA** | Scaled Dot-Product Attention | the fused attention implementation that is avoided on MPS |
| **CLS** | Classification | the `[CLS]` special token whose final vector summarizes a sequence |
| **SEP** | Separator | the `[SEP]` token that separates sequences in a BERT pair input |
| **PAD** | Padding | the `[PAD]` token that makes batch sequences equal length |
| **MT** | Machine Translation | translating text from one language to another |
| **NMT** | Neural Machine Translation | machine translation using neural networks |
| **RNG** | Random Number Generator | the seeded generator used for shuffling and splits |
| **NaN** | Not a Number | an invalid floating-point value; a sign of numerical failure |
| **AdamW** | Adaptive Moment Estimation with decoupled Weight decay | the optimizer used by Hugging Face `Trainer` |
| **SGD** | Stochastic Gradient Descent | the basic optimizer mentioned as a contrast |
| **LR** | Learning Rate | the step size for optimizer updates |

### H.2 Models and translation systems

| Short form | Full form | Meaning in this guide |
| --- | --- | --- |
| **IndicTrans2** | Indic Translation 2 | AI4Bharat's Indic-to-Indic/English translation model family |
| **AI4Bharat** | Artificial Intelligence for Bharat | the research group that released IndicTrans2 |
| **MarianMT** | Marian Machine Translation | the OPUS/Marian neural translation model family |
| **OPUS-MT** | Open Parallel Corpus – Machine Translation | the OPUS collection of pretrained translation models |
| **OPUS** | Open Parallel Corpus | the parallel-text corpus project behind OPUS-MT |
| **NLLB** | No Language Left Behind | Meta's multilingual translation model family, considered as an alternative |
| **IITB** | Indian Institute of Technology Bombay | the source of the Hindi–English evaluation benchmark |
| **HF** | Hugging Face | the model/dataset hosting company and its libraries |
| **HF_TOKEN** | Hugging Face Token | the access token used to download gated models |

### H.3 Evaluation metrics

| Short form | Full form | Meaning in this guide |
| --- | --- | --- |
| **F1** | F1 score | harmonic mean of precision and recall |
| **P** | Precision | fraction of predicted positives that are correct |
| **R** | Recall | fraction of true positives that are found |
| **TP** | True Positive | correctly predicted positive |
| **TN** | True Negative | correctly predicted negative |
| **FP** | False Positive | incorrectly predicted positive |
| **FN** | False Negative | missed positive |
| **BLEU** | Bilingual Evaluation Understudy | n-gram precision metric for translation |
| **BP** | Brevity Penalty | BLEU's penalty for outputs shorter than the reference |
| **chrF** | character n-gram F-score | character-level translation metric |
| **tau** | Kendall rank correlation coefficient (τ) | measures how close a recovered order is to the true order |

### H.4 Hardware, formats, and tooling

| Short form | Full form | Meaning in this guide |
| --- | --- | --- |
| **GPU** | Graphics Processing Unit | the accelerator used for training and inference |
| **CPU** | Central Processing Unit | the fallback processor; much slower for these models |
| **CUDA** | Compute Unified Device Architecture | NVIDIA's GPU programming platform |
| **MPS** | Metal Performance Shaders | Apple's GPU backend used on the M4 |
| **T4** | NVIDIA Tesla T4 | the 16 GB GPU assumed by the Colab recipe |
| **M4** | Apple M4 | the Apple silicon chip used for the verified local run |
| **GB** | Gigabyte | 10^9 bytes (used loosely for GPU memory) |
| **MB** | Megabyte | 10^6 bytes |
| **fp16** | 16-bit floating point | half precision, used on CUDA |
| **fp32** | 32-bit floating point | full precision, used on MPS |
| **OOM** | Out of Memory | the failure when a model exceeds the memory budget |
| **ONNX** | Open Neural Network Exchange | an export format; its removal in transformers 5 is a pin reason |
| **API** | Application Programming Interface | a programmatic interface, e.g. the Hugging Face Hub API |
| **CLI** | Command-Line Interface | the `python -m src...` commands used throughout |
| **JSON** | JavaScript Object Notation | the report and configuration format |
| **JSONL** | JSON Lines | one JSON object per line; the dataset format |
| **HTML** | HyperText Markup Language | the intermediate format used to build the PDF |
| **PDF** | Portable Document Format | this document's format |
| **URL** | Uniform Resource Locator | a web address |
| **URI** | Uniform Resource Identifier | the `file://` address used to load the HTML into Chrome |
| **ID** | Identifier | a unique key, e.g. an article ID or label ID |
| **I/O** | Input/Output | reading and writing data or models |
| **NLTK** | Natural Language Toolkit | the classical NLP library used in the lesson notebooks |
| **Colab** | Google Colaboratory | the hosted notebook environment used by the full recipe |
| **OS** | Operating System | the platform running the code |
| **macOS** | Mac Operating System | Apple's desktop OS used for the verified local run |

### H.5 Licenses and organizations

| Short form | Full form | Meaning in this guide |
| --- | --- | --- |
| **CC BY-SA** | Creative Commons Attribution-ShareAlike | the license of Wikipedia text |
| **GFDL** | GNU Free Documentation License | the other license of Wikipedia text |
| **GNU** | GNU's Not Unix | the project behind the GFDL |

### H.6 Roman numerals and section labels

| Form | Meaning |
| --- | --- |
| **I, II, III, IV** | Part numbers (Part I, Part II, Part III, Part IV) |
| **A.1–A.8** | sections of Part 0.5, the approach chapter |
| **D1–D12** | design decisions in Chapter 28 |
| **H.1–H.7, I.1–I.16** | sections of these appendices |
| **Ch. / §** | chapter / section cross-reference |
| **e.g.** | *exempli gratia* (Latin) | for example |
| **et al.** | *et alia* (Latin) | and others; used in reference citations |
| **README** | “read me” | the repository front page; not an acronym |

### H.7 Symbols and notation

| Form | Meaning |
| --- | --- |
| `[CLS]`, `[SEP]`, `[PAD]` | BERT special tokens (see H.1) |
| `##` | WordPiece continuation marker: this subword continues the previous one |
| `U+0964` | Unicode code point for the Devanagari danda `।` |
| `U+092F`, `U+0927`, `U+093E`, `U+094D` | Unicode code points for DEVANAGARI LETTER YA, DHA, VOWEL SIGN AA, and VIRAMA (`्`) |
| `r`, `alpha`, `dropout` | LoRA rank, scaling factor, and adapter dropout |
| `O(n²)` | quadratic growth in the number of sentences |
| `τ` | Kendall's tau (see H.3) |
| `BP` | BLEU brevity penalty (see H.3) |
| `d_model` / `d_ff` | model hidden dimension / feed-forward hidden dimension |
| `\u0964` | Python escape for the danda `।` |

> **Note.** Unicode character names such as `DEVANAGARI LETTER YA`, `DEVANAGARI LETTER DHA`, `DEVANAGARI VOWEL SIGN AA`, and `VIRAMA` describe code points; they are not abbreviations. All-caps code names such as `LABELS`, `LABEL2ID`, and `PUNCT_DATA` are Python constants, not acronyms. The `##` in `भारत` tokenizations is a WordPiece marker, not a comment.

<div class='pagebreak'></div>

## Appendix I. How to actually test it

This appendix is a practical test plan. It starts with cheap checks and ends with the full pipeline. Every step has a command and a pass condition, so “it works” becomes something you can demonstrate rather than assert.

### I.1 The four test levels

| Level | Question | Typical time | Pass condition |
| --- | --- | --- | --- |
| 0. Environment | do the pinned imports work? | seconds | no import errors; device printed |
| 1. Data | are windows, labels, pairs, and splits correct? | minutes | counts match; no leakage; labels aligned |
| 2. Stage quality | does each model beat its baseline? | minutes | boundary F1 ≥ 0.85; pair accuracy > 0.5; Borda tau > TF-IDF tau |
| 3. End to end | are outputs aligned, complete, and reproducible? | minutes | 4 files per input; equal sentence/group counts; no `NULL` |

### I.2 One-time setup

```bash
source .venv.backup-20261109-124948/bin/activate
python -m compileall -q src
python setup_check.py
```

Expected: `setup_check.py` reports Python, torch, and the available device without raising. On this machine the device is MPS.

### I.3 Test 0 — environment and pins

```bash
python - <<'PY'
import torch, transformers, peft, datasets, sacrebleu
print('torch       ', torch.__version__)
print('transformers', transformers.__version__)
print('peft        ', peft.__version__)
print('datasets    ', datasets.__version__)
print('sacrebleu   ', sacrebleu.__version__)
print('cuda        ', torch.cuda.is_available())
print('mps         ', torch.backends.mps.is_available())
PY
```

Pass condition: the versions match `requirements-colab.txt`, and at least one accelerator is available. The verified local stack is torch 2.15.0, transformers 4.44.2, peft 0.13.2, datasets 3.6.0 on Python 3.12.

### I.4 Test 1 — data integrity

```bash
python -m src.data_prep --num-articles 3000 --max-train-pairs 40000
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
from src.constants import LABELS, MAX_WORDS, MIN_WORDS

root = Path('data/processed')
summary = json.loads((root / 'data_prep_summary.json').read_text())

assert summary['articles'] == 3000, summary['articles']
assert summary['splits'] == {'train': 2700, 'val': 150, 'test': 150}
assert summary['punctuation']['train']['examples'] == 25000
assert summary['ordering']['train']['pairs'] == 40000

windows = {}
for split in ('train', 'val', 'test'):
    rows = [json.loads(line) for line in
            (root / 'punctuation' / f'{split}.jsonl').open(encoding='utf-8')]
    windows[split] = rows
    assert len(rows) == summary['punctuation'][split]['examples']
    for row in rows:
        assert len(row['words']) == len(row['labels'])
        assert MIN_WORDS <= len(row['words']) <= MAX_WORDS
        assert set(row['labels']) <= set(LABELS)

    pairs = [json.loads(line) for line in
             (root / 'ordering' / f'{split}.jsonl').open(encoding='utf-8')]
    assert len(pairs) == summary['ordering'][split]['pairs']
    counts = Counter(pair['label'] for pair in pairs)
    positive_share = counts[1] / len(pairs)
    assert 0.48 <= positive_share <= 0.52, (split, counts)

ids = {split: {row['article_id'] for row in rows}
       for split, rows in windows.items()}
assert not (ids['train'] & ids['val'])
assert not (ids['train'] & ids['test'])
assert not (ids['val'] & ids['test'])

eval_articles = [json.loads(line) for line in
                 (root / 'ordering' / 'ordering_eval.jsonl').open(encoding='utf-8')]
assert len(eval_articles) == 40
assert all(8 <= len(article['sentences']) <= 70 for article in eval_articles)
print('data integrity OK:', {k: len(v) for k, v in windows.items()})
PY
```

Pass conditions: no assertion fires. This proves that words and labels align, windows respect the 32–96 word bounds, ordering pairs are approximately balanced after capping (49–50% positive), and no article appears in two splits.

### I.5 Test 2 — punctuation adapter

```bash
python - <<'PY'
import json
from pathlib import Path
from src.punctuation.predict import PunctuationRestorer

row = json.loads(next((Path('data/processed/punctuation/test.jsonl')
                       .open(encoding='utf-8'))))
words = row['words']
restorer = PunctuationRestorer('models/punctuation', memory_gb=1.5)
labels, punctuated, plain = restorer.restore(words)

assert len(labels) == len(words)
assert all(0 <= label < 5 for label in labels)
assert len(punctuated) == len(plain) >= 1
assert all(item.strip() for item in punctuated)
first = restorer.predict_labels(words)
second = restorer.predict_labels(words)
assert first == second, 'inference is not deterministic'
restorer.close()
print('punctuation OK:', len(words), 'words,', len(punctuated), 'sentences')
PY
```

Pass condition: the output lengths match, labels are in range, sentences are non-empty, and two calls return identical labels. The model is in `eval()` mode and uses argmax, so determinism is expected.

### I.6 Test 3 — ordering adapter and decoders

```bash
python - <<'PY'
import json
from pathlib import Path
from src.ordering.decode import OrderScorer, order_sentences

article = json.loads(next((Path('data/processed/ordering/ordering_eval.jsonl')
                           .open(encoding='utf-8'))))
sentences = article['sentences'][:8]
scorer = OrderScorer('models/ordering', memory_gb=1.5)
matrix = scorer.score_matrix(sentences)

assert matrix.shape == (len(sentences), len(sentences))
assert all(abs(matrix[i, i] - 0.5) < 1e-9 for i in range(len(sentences)))
for i in range(len(sentences)):
    for j in range(len(sentences)):
        assert abs(matrix[i, j] + matrix[j, i] - 1.0) < 1e-6

for strategy in ('borda', 'clusters', 'beam'):
    order = order_sentences(sentences, scorer, strategy=strategy)
    assert sorted(order) == list(range(len(sentences))), strategy
scorer.close()
print('ordering OK: 8 sentences, 3 decoders produced valid permutations')
PY
```

Pass condition: the matrix is symmetric-complementary, the diagonal is 0.5, and every decoder returns a true permutation (no duplicates, no missing indices).

### I.7 Test 4 — translation adapter

```bash
python - <<'PY'
from src.translation.indictrans import IndicTrans2Translator

translator = IndicTrans2Translator(
    'models/translation/indictrans2-indic-en-200M', memory_gb=1.5)
texts = ['भारत एक विशाल देश है।', 'यह एक परीक्षण है।', 'समय बहुत महत्वपूर्ण है।']
outputs = translator.translate(texts, batch_size=2)
translator.close()

assert len(outputs) == len(texts)
assert all(output.strip() for output in outputs)
assert all(any(ch.isalpha() for ch in output) for output in outputs)
print('translation OK:')
for hindi, english in zip(texts, outputs):
    print(' ', hindi, '->', english)
PY
```

Pass condition: one non-empty English string per Hindi input. The exact wording can vary with beams or model version; do not assert exact strings.

### I.8 Test 5 — end-to-end artifacts

```bash
python -m src.pipeline \
    --translation-model models/translation/indictrans2-indic-en-200M \
    --translation-batch-size 8 \
    --output-dir /tmp/pipeline-check

python - <<'PY'
import math
from pathlib import Path

root = Path('/tmp/pipeline-check')
expected = {
    'sample-1': (52, 11), 'sample-2': (55, 11), 'sample-3': (49, 10),
    'sample-4': (53, 11), 'sample-5': (55, 11), 'sample-6': (80, 16),
    'sample-7': (99, 20),
}
for stem, (sentences, groups) in expected.items():
    hi = (root / f'{stem}.sentences_hi.txt').read_text(encoding='utf-8').splitlines()
    en = (root / f'{stem}.sentences_en.txt').read_text(encoding='utf-8').splitlines()
    ghi = (root / f'{stem}.groups_hi.txt').read_text(encoding='utf-8').splitlines()
    gen = (root / f'{stem}.groups_en.txt').read_text(encoding='utf-8').splitlines()

    assert len(hi) == sentences, (stem, len(hi))
    assert len(en) == len(hi), stem
    assert len(ghi) == len(gen) == groups == math.ceil(len(hi) / 5), stem
    assert all(line.strip() for line in hi + en + ghi + gen)
    assert not any('NULL' in line for line in hi), stem
    assert any('\u0900' <= char <= '\u097f' for char in ''.join(hi)), stem
    assert any(char.isascii() and char.isalpha() for char in ''.join(en)), stem
print('end-to-end OK: all seven files have aligned Hindi/English outputs')
PY
```

Pass conditions: all assertions hold. These are the same counts reported in Chapter 2 and `reports/run_summary.json`.

### I.9 Test 6 — failure behavior

A good pipeline fails loudly. Run these and confirm a non-zero exit and a clear message:

```bash
# Missing adapter
python -m src.pipeline --limit 1 --punctuation-model /tmp/does-not-exist
# Expect: FileNotFoundError ... Train it first

# Missing input directory
python -m src.pipeline --input-dir /tmp/no-such-dir
# Expect: SystemExit with the directory path

# Missing translation model
python -m src.pipeline --limit 1 --translation-model /tmp/no-such-model
# Expect: an OSError/ImportError with the model path
```

Pass condition: none of these silently falls back to a different model or writes empty outputs. That is a deliberate design choice.

### I.10 Test 7 — determinism

```bash
python -m src.pipeline --limit 1 --output-dir /tmp/run-a
python -m src.pipeline --limit 1 --output-dir /tmp/run-b
diff -r /tmp/run-a /tmp/run-b
```

Pass condition: `diff` prints nothing. Timings differ, but they are written to `reports/run_summary.json`, not to the per-file outputs. Do not compare `run_summary.json` itself across runs.

### I.11 Test 8 — the executed notebook

```bash
jupyter nbconvert --to notebook --execute notebooks/pipeline_demo.ipynb \
    --output /tmp/demo.executed.ipynb --ExecutePreprocessor.timeout=900
python - <<'PY'
import nbformat
nb = nbformat.read('/tmp/demo.executed.ipynb', as_version=4)
errors = [out for cell in nb.cells for out in cell.get('outputs', [])
          if out.get('output_type') == 'error']
assert not errors, errors
print('notebook OK:', len(nb.cells), 'cells, no error outputs')
PY
```

Pass condition: the notebook executes and contains no error outputs. The checked-in notebook is already executed; this test re-runs it from scratch.

### I.12 Test 9 — memory budget

```bash
python - <<'PY'
import torch
from src.utils import get_device, limit_gpu_memory
print('device:', get_device())
print('cap:', limit_gpu_memory(1.5))
if torch.cuda.is_available():
    print('allocated GB:', torch.cuda.memory_allocated() / 1e9)
PY
```

Pass condition: `limit_gpu_memory(1.5)` succeeds, and a full pipeline run does not raise a CUDA out-of-memory error. The cap is enforced by `torch.cuda.set_per_process_memory_fraction`, so exceeding it fails rather than silently using more. On MPS the function is a no-op; peak memory is reported by the training summaries instead.

### I.13 Test 10 — report consistency

```bash
python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path('reports/evaluation.json').read_text())
assert report['no_llm'] is True
assert report['memory_limit_gb'] == 1.5
assert report['punctuation']['status'] == 'ok'
assert report['punctuation']['boundary']['f1'] >= 0.85
assert report['punctuation']['accuracy'] >= 0.85
assert report['ordering']['pairs']['accuracy'] > 0.5
assert report['ordering']['kendall_tau']['borda']['mean_tau'] > \
       report['ordering']['kendall_tau']['tfidf_ablation']['mean_tau']
assert report['translation']['status'] == 'ok'
assert report['translation']['bleu'] >= 30.0
print('report OK:', report['punctuation']['boundary']['f1'],
      report['ordering']['pairs']['accuracy'],
      report['ordering']['kendall_tau']['borda']['mean_tau'],
      report['translation']['bleu'])
PY
```

Pass condition: no assertion fires. The thresholds are deliberately below the measured values, so they detect regressions without failing on harmless variance.

### I.14 Expected values for the verified local run

| Check | Expected |
| --- | --- |
| Punctuation test windows | 2,208 |
| Punctuation evaluation words | 108,898 |
| Punctuation accuracy | 0.8724 |
| Punctuation boundary F1 | 0.8842 |
| Ordering test pair accuracy | 0.6543 |
| Ordering Borda mean tau | 0.2007 |
| Ordering TF-IDF mean tau | 0.0631 |
| Translation BLEU (100 IITB sentences) | 34.125 |
| Translation chrF | 59.686 |
| Sentences, sample-1 to sample-7 | 52, 55, 49, 53, 55, 80, 99 |
| Groups, sample-1 to sample-7 | 11, 11, 10, 11, 11, 16, 20 |
| Pipeline timings (MPS) | 2.31 s / 130.53 s / 102.17 s |

### I.15 Testing on your own data

The sample files are only seven inputs. To test whether the pipeline generalizes, use a new Hindi text and one of the two paths below.

**Path A: unlabeled custom input (does it run and look sane?).**

The pipeline expects a `.txt` file of whitespace-separated Hindi words with no punctuation. `NULL` tokens are allowed and removed. If you start from a normal punctuated article, flatten it first:

```bash
mkdir -p data/custom
python - <<'PY'
import random
import re
from pathlib import Path
from src.utils import strip_punctuation

source = Path('my_article_raw.txt')          # your punctuated Hindi text
output = Path('data/custom/my_article.txt')
text = re.sub(r'\[\d+\]', '', source.read_text(encoding='utf-8'))
parts = re.split(r'(?<=[।॥?!])\s+', text)

blocks = []
for part in parts:
    words = [strip_punctuation(token)[0] for token in part.split()]
    words = [word for word in words if word]
    if words:
        blocks.append(words)

random.Random(42).shuffle(blocks)             # mimic shuffled assignment input
flat = ' '.join(word for block in blocks for word in block)
output.write_text(flat, encoding='utf-8')
print('wrote', output, 'with', len(flat.split()), 'words')
PY
```

Then run only the Hindi stages, or the full pipeline:

```bash
# Fast: punctuation + ordering only
python -m src.pipeline --input-dir data/custom --output-dir outputs/custom \
    --no-translate

# Full pipeline
python -m src.pipeline --input-dir data/custom --output-dir outputs/custom \
    --translation-model models/translation/indictrans2-indic-en-200M \
    --translation-batch-size 8
```

Check the four outputs and their alignment:

```bash
wc -l outputs/custom/my_article.sentences_hi.txt \
      outputs/custom/my_article.sentences_en.txt \
      outputs/custom/my_article.groups_hi.txt \
      outputs/custom/my_article.groups_en.txt
```

Pass condition: all four files exist, Hindi and English sentence counts match, and group counts match `ceil(sentences / 5)`. With no references this proves execution and plausibility, not accuracy.

**Path B: labeled custom evaluation (does it generalize numerically?).**

Create JSONL in the same schemas used by `src.evaluate`. For punctuation, one object per window:

```json
{"id": "custom-1", "article_id": "custom-article-1", "split": "test",
 "words": ["भारत", "एक", "विशाल", "देश", "है"],
 "labels": ["none", "none", "none", "none", "sentence_end"]}
```

For ordering pairs, one object per pair:

```json
{"sentence_a": "पहला वाक्य है।", "sentence_b": "दूसरा वाक्य है।", "label": 1}
```

For full-order evaluation, one object per held-out article, in the true order:

```json
{"article_id": "custom-article-1",
 "sentences": ["पहला वाक्य है।", "दूसरा वाक्य है।", "तीसरा वाक्य है।"]}
```

Then run:

```bash
python -m src.evaluate \
    --punct-test data/custom/punct_test.jsonl \
    --skip-ordering --skip-translation --report /tmp/custom-punct.json

python -m src.evaluate \
    --order-test data/custom/order_test.jsonl \
    --order-eval data/custom/order_eval.jsonl \
    --skip-punctuation --skip-translation --report /tmp/custom-order.json
```

Pass condition: the report contains `status: ok` and a metric for your data. Keep the custom article IDs disjoint from any training data; otherwise the score is leakage, not generalization.

**Translation on your own parallel data.** `src.evaluate` streams the IITB benchmark, so for your own Hindi–English pairs use the wrapper directly:

```bash
python - <<'PY'
import json
import sacrebleu
from src.translation.indictrans import IndicTrans2Translator

pairs = [json.loads(line) for line in
         open('data/custom/parallel.jsonl', encoding='utf-8')]
translator = IndicTrans2Translator(
    'models/translation/indictrans2-indic-en-200M', memory_gb=1.5)
hypotheses = translator.translate([pair['hindi'] for pair in pairs], batch_size=8)
translator.close()

references = [[pair['english'] for pair in pairs]]
print('BLEU:', sacrebleu.corpus_bleu(hypotheses, references).score)
print('chrF:', sacrebleu.corpus_chrf(hypotheses, references).score)
PY
```

The input file has one object per line: `{"hindi": "...", "english": "..."}`.

**What to expect.** These models learned Wikipedia prose. News and encyclopedia text should behave similarly; dialogue, poetry, song lyrics, and social-media text will be worse, especially for sentence boundaries and ordering. Report the domain next to the score; a metric on out-of-domain text is not comparable to the Wikipedia test set.

### I.16 What to show a reviewer

1. `reports/evaluation.json` — every stage `status: ok`, with the metrics above.
2. `reports/run_summary.json` — the four output counts per file and the timings.
3. The four output files for one sample — aligned Hindi and English, groups of five.
4. `notebooks/pipeline_demo.ipynb` — executed, with visible outputs and no errors.
5. `models/*/training_summary.json` — trainable parameters, validation/test metrics, peak memory.
6. The failure tests in I.9 — proof that the pipeline does not hide problems.

> **The honest claim.** “On this machine, with 4,000-example subset adapters and a local IndicTrans2 mirror, the pipeline runs end to end, the stage metrics pass these thresholds, and the outputs align. The full-data recipe is documented but was not run to completion here.” That sentence is defensible; anything stronger is not supported by the artifacts.
