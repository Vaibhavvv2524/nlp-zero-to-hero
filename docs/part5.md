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
<strong>APPROACH CHOSEN.</strong> Make each major decision explicit, with the alternative, the reason, and the cost. This is the material a reviewer or examiner will probe, so it is written as a decision log rather than prose.
</div>

### D1. Language: Hindi

- **Chosen:** Hindi (Devanagari), with English translation.
- **Alternatives:** a higher-resource language with better tooling.
- **Why:** the task requires Hindi; Devanagari has clear sentence-ending danda, rich morphology, and available multilingual models.
- **Cost:** fewer high-quality tools than English; tokenization is harder.

### D2. Tokenization: whitespace, not regex

- **Chosen:** `str.split()`.
- **Alternatives:** `re.findall(r'\w+', text)`.
- **Why:** Devanagari vowel signs are combining marks; `\w+` drops them (`अयोध्या` → `अय`, `ध`, `य`). Whitespace keeps words intact.
- **Cost:** punctuation attached to words must be handled by the models or stripped.

### D3. Punctuation: token classification, not seq2seq

- **Chosen:** classify each word into five labels.
- **Alternatives:** generate punctuated text with a seq2seq model; use a large LLM.
- **Why:** one-to-one alignment, fast, small, no hallucination, fits 1.5 GB.
- **Cost:** cannot insert words; rare labels are hard; comma style is limited.

### D4. Model: mBERT + LoRA

- **Chosen:** `bert-base-multilingual-cased` with LoRA r=16 on query/value.
- **Alternatives:** full fine-tuning; a larger multilingual model; XLM-R; MuRIL.
- **Why:** mBERT has Hindi coverage, is ~178M params, and LoRA makes training cheap. It was available locally.
- **Cost:** mBERT is older than XLM-R/MuRIL; cased vocab splits Devanagari heavily.

### D5. Label alignment: first subword

- **Chosen:** assign the word label to its first subword; ignore the rest.
- **Alternatives:** average subword logits; predict on the last subword; use word-level tokenizer.
- **Why:** Bengali/Hindi word endings carry punctuation cues, but first-subword is simple, stable, and works well in practice.
- **Cost:** loses some information from later subwords; a learned pooling could be better.

### D6. Ordering: pairwise binary classification + Borda

- **Chosen:** train `[CLS] A [SEP] B [SEP] → before/after`; decode with Borda.
- **Alternatives:** pointer networks; seq2seq ordering; listwise ranking; topological sort.
- **Why:** pairwise data is easy to generate and the classifier is small and robust. Borda aggregates noisy preferences.
- **Cost:** pairwise preferences can be cyclic; Borda is not optimal; full order is weak.

### D7. Ordering training data: balanced both directions

- **Chosen:** for each adjacent pair, add `(A,B,1)` and `(B,A,0)`.
- **Alternatives:** only natural order; sample random pairs.
- **Why:** removes position bias and doubles data.
- **Cost:** model cannot use absolute position; only relative coherence.

### D8. Translation: IndicTrans2 200M

- **Chosen:** `ai4bharat/indictrans2-indic-en-dist-200M`.
- **Alternatives:** MarianMT; NLLB; a large LLM; Google Translate API.
- **Why:** Indic-specialized, 200M is feasible, strong Hindi→English quality, runs locally.
- **Cost:** gated license; ~913 MB; 256-token limit; slower on CPU.

### D9. Memory: staged load/free vs simultaneous

- **Chosen:** load one model, run its stage on all files, free it, load the next.
- **Alternatives:** load all three and stream.
- **Why:** three models would exceed 1.5 GB; staging caps peak memory at the largest model.
- **Cost:** cannot jointly optimize stages; more I/O.

### D10. Evaluation: per-stage, honest subsets

- **Chosen:** held-out windows/pairs/articles and a 100-sentence translation sample.
- **Alternatives:** one end-to-end human evaluation.
- **Why:** fast, reproducible, isolates failures.
- **Cost:** does not measure full-pipeline quality directly.

### D11. Notebooks: recipe + executed demo

- **Chosen:** a full Colab recipe notebook and an executed local demo.
- **Alternatives:** only scripts; only a notebook.
- **Why:** the recipe is portable; the demo is evidence.
- **Cost:** two artifacts to maintain; the recipe is not executed here.

### D12. No LLM

- **Chosen:** no generative LLM anywhere in the pipeline.
- **Alternatives:** prompt GPT/Gemini for punctuation/ordering/translation.
- **Why:** the assignment forbids an LLM-based solution; also cost, latency, privacy, and determinism.
- **Cost:** lower ceiling on quality, especially ordering.

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

`reports/data_prep_summary.json`:

```json
{
  "articles": {"train": 2700, "val": 150, "test": 150},
  "punctuation": {"train": 25000, "val": 2317, "test": 2208},
  "ordering": {"train": 40000, "val": 3000, "test": 3000,
               "eval_articles": 40},
  "label_counts": {"none": 1653107, "comma": 59324,
                   "sentence_end": 104840,
                   "question": 236, "exclamation": 153}
}
```

`models/*/training_summary.json`:

```json
{
  "base_model": "bert-base-multilingual-cased",
  "trainable_parameters": 593669,
  "total_parameters": 177860362,
  "train_metrics": {},
  "eval_metrics": {"eval_loss": 0.4399, "eval_boundary_f1": 0.8246},
  "test_metrics": {"test_accuracy": 0.8688, "test_boundary_f1": 0.8633},
  "peak_memory_gb": 0.0
}
```

`reports/run_summary.json`:

```json
{
  "strategy": "borda",
  "timings": {"punctuation_seconds": 2.31,
              "ordering_seconds": 130.53,
              "translation_seconds": 102.17},
  "files": {"sample-1": {"words": 747, "null_removed": 5,
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
