# 0. Document contract

This document is the **approach dossier** for the Hindi text-restoration project. It is deliberately not a beginner tutorial and not a general NLP textbook. It answers one question, exhaustively:

> **What is the approach, how exactly does it work, why was it chosen, what does it cost, what does it fail at, and what is the evidence?**

The document is organised around the design rather than the code. Every claim is tied to a constraint, a source file, a measured artifact, or an explicitly labelled assumption.

<div class="evidence">

**Evidence labels used throughout**

- **Measured locally** — produced on this machine by the checked-in adapters and reports.
- **Documented recipe** — the full-data path exists in code but was not run to completion here.
- **Design rationale** — a reasoned choice supported by constraints and code, not a measured benchmark.
- **Known limitation** — an observed or structural weakness, stated rather than hidden.

</div>

## 0.1 Claim boundary

The repository contains two different things: a **full-data recipe** and a **measured local subset run**. They must never be conflated.

| Item | Full-data recipe | Measured local run |
| --- | --- | --- |
| Punctuation training data | 25,000 windows | 4,000 windows |
| Ordering training data | 40,000 pairs | 4,000 pairs |
| Hardware | Colab T4 / CUDA | Apple M4 / MPS |
| Memory cap | 1.5 GB CUDA fraction enforced | MPS; CUDA cap not exercised |
| Translation model | IndicTrans2 200M | local IndicTrans2 mirror |
| Status | documented, not executed here | executed, with saved reports |
| Metrics | expected to be stronger | reported in this document |

The local adapters are **subset adapters**. Their scores are honest for the subset run, but they are not the scores of the full 25,000/40,000 recipe. The doc uses the local numbers wherever a number is required and marks the full-data claims as expectations, not results.

## 0.2 Notation and abbreviations

| Symbol / term | Meaning |
| --- | --- |
| $w_1,\ldots,w_n$ | input words after `NULL` removal |
| $s_1,\ldots,s_m$ | sentences produced by stage 1 |
| $L$ | number of punctuation labels, five |
| $p_i$ | probability distribution over the five punctuation labels for word $i$ |
| $M_{ij}$ | model probability that sentence $i$ comes before sentence $j$ |
| $n$ | number of sentences in one ordering problem |
| $\tau$ | Kendall's rank correlation, in $[-1,1]$ |
| Borda | average-preference ranking rule used by the default decoder |
| LoRA | low-rank adaptation of frozen model weights |
| mBERT | `bert-base-multilingual-cased` |
| IndicTrans2 | `ai4bharat/indictrans2-indic-en-dist-200M` |
| boundary | `sentence_end`, `question`, or `exclamation` |
| group | five consecutive output sentences |
| subset adapter | adapter trained on 4,000 examples rather than the full recipe |

## 0.3 Source-of-truth map

| Area | Primary source |
| --- | --- |
| Paths, labels, model ids, window sizes | `src/constants.py` |
| Dataset construction | `src/data_prep.py`, `src/ordering/pairs.py` |
| Punctuation training and inference | `src/punctuation/train.py`, `dataset.py`, `predict.py` |
| Ordering training and decoding | `src/ordering/train.py`, `decode.py` |
| Translation wrapper | `src/translation/indictrans.py` |
| Orchestration and output contract | `src/pipeline.py`, `src/utils.py` |
| Evaluation | `src/evaluate.py` |
| Measured artifacts | `reports/*.json`, `outputs/*.txt` |
| Automated checks | `scripts/verify_colab_run.py` |

## 0.4 Reading order

1. Sections 1–4 give the thesis, constraints, and system contract.
2. Sections 5–8 explain each stage in depth.
3. Sections 9–12 cover memory, training, evaluation, and failure analysis.
4. Sections 13–15 are the decision log, pros/cons, and improvement plan.
5. Sections 16–20 cover evidence, implementation, operations, review questions, and references.

<div class="pagebreak"></div>

# 1. Executive summary

The project takes a flat, unpunctuated, shuffled Hindi token stream and produces aligned Hindi and English sentences, grouped five at a time. The approach is a **three-stage decomposition**:

1. **Punctuation** — mBERT with a LoRA adapter labels every word as `none`, `comma`, `sentence_end`, `question`, or `exclamation`. Boundary labels are used to rebuild sentences.
2. **Ordering** — a second mBERT + LoRA model scores whether sentence A comes before sentence B. A separate decoder turns the pairwise score matrix into one global order.
3. **Translation** — IndicTrans2 200M translates each ordered Hindi sentence into English. Sentence-level translation preserves exact alignment and the group structure.

The pipeline loads exactly one model at a time, uses fp16 on CUDA, and relies on LoRA plus gradient checkpointing to stay inside the 1.5 GB assignment budget. There is no LLM and no fallback translator.

## 1.1 The approach in one sentence

> **Label words, rank sentences in pairs, decode a global order, translate sentence by sentence, and keep every intermediate structure aligned.**

That sentence is the whole design. The rest of this document explains why each verb is the right one for this problem.

## 1.2 The three stages at a glance

| Stage | Input | Model | Output | Decoder |
| --- | --- | --- | --- | --- |
| 1. Punctuation | flat Hindi word list | mBERT + LoRA, 5-class token classifier | one label per word; sentence strings | argmax over triangular-weighted overlapping windows |
| 2. Ordering | list of plain Hindi sentences | mBERT + LoRA, 2-class pair classifier | full sentence permutation | Borda by default; clusters or beam optionally |
| 3. Translation | ordered Hindi sentences | IndicTrans2 200M encoder-decoder | aligned English sentences | beam search, 4 beams |

## 1.3 The eight principles

<div class="principles">

**1. Preserve the words; predict structure.** Stage 1 attaches labels to existing words instead of regenerating text, so the word stream is auditable and exact word-level evaluation is possible.

**2. Learn from natural supervision.** Wikipedia already contains punctuation and true sentence order. The project hides those signals to create targets instead of asking an LLM to generate data.

**3. Split by article, shuffle inside windows.** No article appears in two splits. Shuffling sentences inside punctuation windows prevents the model from using cross-sentence continuity that will not exist at inference.

**4. Reuse one backbone.** mBERT serves both learned Hindi stages. This keeps the model surface small and the memory profile predictable.

**5. Adapt with LoRA.** Only about 0.33% of parameters are trainable. The frozen backbone and tiny adapters make the 1.5 GB target plausible and the artifacts small.

**6. Separate learning from decoding.** The ordering model answers only pairwise questions. Borda, clustering, or beam search turns those answers into a global order without retraining.

**7. Translate sentence by sentence.** IndicTrans2 is used one sentence at a time so Hindi line $i$ maps to English line $i$ and the groups of five remain valid.

**8. Stage memory and fail loudly.** Load one model, run its stage, free it. If an adapter or translator is missing, raise an error; never silently substitute another model.

</div>

## 1.4 Headline measured results

These are the local subset-adapter numbers. They are reproduced from `reports/evaluation.json` and `reports/run_summary.json`.

| Metric | Result | What it means |
| --- | ---: | --- |
| Punctuation token accuracy | 0.8724 | high partly because `none` is 91% of tokens |
| Punctuation boundary F1 | 0.8842 | the meaningful punctuation score |
| Comma F1 | 0.3161 | comma placement is the hard class |
| Question / exclamation F1 | 0.0000 / 0.0000 | too few training examples |
| Ordering pair accuracy | 0.6543 | clearly above 0.5, but not enough for full order |
| Ordering macro F1 | 0.6534 | balanced pair performance |
| Borda mean Kendall's tau | 0.2007 | full-document order remains weak |
| TF-IDF ablation tau | 0.0631 | Borda beats the lexical baseline |
| Translation BLEU | 34.125 | 100 IITB Hindi-English pairs |
| Translation chrF | 59.686 | character-level overlap |

The honest summary is: **stage 1 works well on sentence endings; stage 2 is directionally useful but not solved; stage 3 is strong on clean sentence pairs.** The architecture is sound; the local subset adapters are the limiting factor.

## 1.5 What the approach is not

- It is **not** an end-to-end generative model.
- It is **not** an LLM-based rewriting system.
- It is **not** a full-document translation system.
- It is **not** a claim that ordering is solved.
- It is **not** a claim that the 1.5 GB CUDA cap was measured on this machine.

Those exclusions are part of the design, not omissions. The next section explains the constraints that forced them.

<div class="pagebreak"></div>
