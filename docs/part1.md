# NLP Zero to Hero

## Hindi Text Restoration and Translation — the complete study guide

This guide explains the project from first principles: what NLP is, how text becomes tokens and vectors, how transformers and LoRA work, how each of the three assignment stages is built, what every source file does, what the notebooks do cell by cell, what the measured results mean, and how to defend the work in a viva.

It is written for someone who is starting from zero but wants to finish with a genuinely deep understanding. Nothing important is hidden behind “it just works”.

> **Scope of the claims in this guide.** All numbers come from the JSON reports and training summaries in this repository. The punctuation and ordering adapters that produced the reported pipeline outputs were trained on 4,000 punctuation windows (2 epochs) and 4,000 ordering pairs (1 epoch) on Apple MPS. The full-data Colab recipe (25,000 windows, 40,000 pairs) is documented but was not completed locally. Wherever a number depends on that distinction, the guide says so explicitly.

<div class='pagebreak'></div>

# Part 0 — Orientation

## 1. The assignment in plain English

You are given Hindi text with three problems:

1. **No punctuation and no sentence boundaries.** The words arrive in a flat stream. You must decide where commas, full stops (the Devanagari danda `।`), question marks, and exclamation marks belong.
2. **The sentences are shuffled.** Even after you recover the boundaries, the sentences are not in their original order. You must recover a sensible order.
3. **The text must be translated into English**, and the translation must preserve the grouping.

The required deliverables are, for each input file: the Hindi sentences in recovered order, the same sentences grouped into blocks of five, the English translation of each sentence, and the English groups.

A hard constraint is attached: **no LLM may be used**, and the whole pipeline must fit in a **1.5 GB GPU memory budget**. That is why the project uses two small, task-specific model families rather than an instruction-following chat model.

The final architecture is:

| Stage | Question answered | Model | Output |
| --- | --- | --- | --- |
| 1. Punctuation | Does this word end a sentence, take a comma, or take no mark? | `bert-base-multilingual-cased` + LoRA token classifier | one label per word |
| 2. Ordering | Given sentences A and B, does A come before B? | `bert-base-multilingual-cased` + LoRA pair classifier | probability per pair, decoded into a full order |
| 3. Translation | What is the English for this Hindi sentence? | `ai4bharat/indictrans2-indic-en-dist-200M` | English sentence |

> **Key idea.** The project is not one model. It is three models plus a data pipeline, glued together by a small amount of careful engineering. Most of the difficulty is in the data representation, the label alignment, and the decoding logic, not in the model code.

## 2. The seven inputs and the four outputs

The raw inputs live in `data/raw/sample-1.txt` through `sample-7.txt`. Each file is space-separated tokens. The assignment's tokenizer has replaced some words with the literal token `NULL`; those must be removed before modelling. The files contain no punctuation and no sentence order information.

After `python -m src.pipeline`, each input produces four files:

| Output file | What it contains |
| --- | --- |
| `outputs/sample-N.sentences_hi.txt` | one punctuated Hindi sentence per line, in recovered order |
| `outputs/sample-N.groups_hi.txt` | one line per group of five Hindi sentences |
| `outputs/sample-N.sentences_en.txt` | one English translation per Hindi sentence, same order |
| `outputs/sample-N.groups_en.txt` | English translations joined into the same groups of five |

The produced counts are:

| File | Words after removing `NULL` | `NULL` removed | Recovered sentences | Groups of five |
| --- | ---: | ---: | ---: | ---: |
| sample-1 | 747 | 5 | 52 | 11 |
| sample-2 | 768 | 5 | 55 | 11 |
| sample-3 | 742 | 6 | 49 | 10 |
| sample-4 | 1,118 | 4 | 53 | 11 |
| sample-5 | 1,026 | 9 | 55 | 11 |
| sample-6 | 507 | 11 | 80 | 16 |
| sample-7 | 846 | 27 | 99 | 20 |

The last group in a file is allowed to contain fewer than five sentences. Grouping is purely positional: sentences 1-5 become group 1, sentences 6-10 become group 2, and so on.

> **Why sample-6 and sample-7 have so many sentences.** Those files contain dialogue-like text with many short utterances. The punctuation model learned sentence endings from Wikipedia, where complete sentences are usually at least five words long. Short utterances therefore look like sentence endings to the model, and the file is over-segmented. This is a real limitation, not a bug in the grouping code.

## 3. The big picture: data flow and model flow

```text
data/raw/sample-N.txt
        |
        |  read_tokens(): str.split(), drop NULL
        v
flat list of Hindi words
        |
        |  Stage 1: PunctuationRestorer
        |  - 64-word sliding windows, stride 40
        |  - mBERT + LoRA token classifier
        |  - triangular weighted averaging
        v
one label per word  +  punctuated sentences  +  plain sentences
        |
        |  Stage 2: OrderScorer
        |  - all sentence pairs [CLS] A [SEP] B [SEP]
        |  - P(A before B) for every pair
        |  - Borda / clusters / beam decoding
        v
recovered sentence order (a permutation of the sentence indices)
        |
        |  Stage 3: IndicTrans2Translator
        |  - IndicProcessor tags + preprocessing
        |  - 200M encoder-decoder, beam search
        v
English sentences in the same order
        |
        |  group_lines(..., 5)
        v
outputs/sample-N.{sentences_hi,groups_hi,sentences_en,groups_en}.txt
```

A crucial engineering decision is that the three models are **never resident in memory at the same time**. `src/pipeline.py` loads stage 1, uses it for every file, closes it, loads stage 2, and so on. This is what makes the 1.5 GB budget plausible on a T4: the budget applies to the largest single stage, not to the sum of all three.

## 4. Vocabulary for the rest of the guide

| Term | One-line meaning |
| --- | --- |
| **Token** | The unit a model reads: a word, a subword, a character, or a special symbol. |
| **Tokenization** | Splitting text into tokens. |
| **Subword** | A piece of a word, used so the model can handle rare words. |
| **Embedding** | A dense vector representing a token, word, or sentence. |
| **Transformer** | A neural architecture built from attention layers. |
| **BERT** | A transformer encoder pretrained to fill in masked words; good at understanding text. |
| **mBERT** | The multilingual version of BERT, pretrained on many languages including Hindi. |
| **LoRA** | Low-Rank Adaptation: train small matrices instead of all model weights. |
| **Logit** | The raw, unnormalized score a model produces before softmax. |
| **Softmax** | Turns logits into probabilities that sum to one. |
| **Fine-tuning** | Continuing to train a pretrained model on a task-specific dataset. |
| **Epoch** | One full pass over the training set. |
| **Batch** | A group of examples processed together. |
| **Cross-entropy** | The standard classification loss; punishes confident wrong answers. |
| **Inference** | Using a trained model to make predictions. |
| **Decoder** | The component that turns model scores into a final structure (a label, an order, a translation). |
| **BLEU / chrF** | Metrics for machine translation quality. |
| **Kendall's tau** | A rank-correlation metric for how close a recovered order is to the truth. |

<div class='pagebreak'></div>

# Part 0.5 — The approach we chose (read this first)

<div class='approach'>
<strong>APPROACH CHOSEN — the thesis of the whole project.</strong><br>
We deliberately <em>decomposed</em> the assignment into three specialized stages instead of asking one model to do everything: a word-level punctuation labeller, a pairwise sentence-order scorer with an explicit decoder, and a dedicated Indic-to-English translator. The first two stages share one small pretrained backbone (mBERT) and are adapted with LoRA; the third is IndicTrans2 200M. There is no LLM, no fallback model, and no stage is allowed to rewrite the input words. The pipeline is engineered around the 1.5 GB budget by loading exactly one model at a time.
</div>

This chapter is intentionally placed before the theory. If you understand *why* this architecture was chosen, every later chapter has a place to hang. The rest of the guide explains the machinery; this chapter explains the design.

## A.1 Why decompose the problem at all?

A single end-to-end model could, in principle, read the flat token stream and emit punctuated, ordered English. We rejected that for four reasons:

| Reason | Explanation |
| --- | --- |
| **Constraint fit** | A single generative model large enough to do all three tasks well would not fit 1.5 GB and would look like an LLM, which the assignment forbids. |
| **Evaluation** | Decomposition gives one metric per stage: boundary F1, pair F1/Kendall's tau, BLEU/chrF. A single model gives one opaque score, and a failure is hard to localize. |
| **Data availability** | Wikipedia gives us punctuation labels and natural sentence order for free. It does not give us aligned triplets of (unpunctuated shuffled Hindi, punctuated ordered Hindi, English). Decomposition lets us train each stage on the supervision that actually exists. |
| **Failure isolation** | If ordering is weak, punctuation and translation can still be correct and measurable. If translation is unavailable, the first two stages still produce a valid partial result. |

The decomposition is not an accident of convenience. It is the design that makes the constraints, the data, and the evaluation line up.

## A.2 The eight principles behind the chosen approach

<div class='principles'>

**1. Preserve the words; predict structure.** Punctuation is a labelling problem, not a rewriting problem. Stage 1 attaches one of five labels to each existing word. This guarantees that the output words are the input words (minus `NULL`) and makes word-level evaluation exact.

**2. Learn from natural text, not synthetic text.** The training corpus is Hindi Wikipedia. The punctuation and sentence order are already there; we only hide them to create supervision. This avoids hand-written rules and avoids LLM-generated data.

**3. Split by article; shuffle inside windows.** The same article never appears in both training and test. Inside a punctuation window, sentences are shuffled to mimic the assignment input and to stop the model from leaning on cross-sentence continuity.

**4. Reuse one backbone for two tasks.** mBERT is used once as a token classifier and once as a pair classifier. This keeps the dependency surface small, the memory profile predictable, and the story simple: one multilingual encoder, two heads.

**5. Adapt cheaply with LoRA.** Full fine-tuning of 178M parameters is unnecessary and risky under the budget. LoRA trains about 0.33% of the parameters, keeps optimizer state tiny, and produces adapters small enough to save and share.

**6. Make local decisions, then decode globally.** The ordering model answers only “does A come before B?”. A separate decoder (Borda by default) turns all pairwise probabilities into a global permutation. This separates learning from combinatorial optimization and lets us compare decoders without retraining.

**7. Use a dedicated translator, sentence by sentence.** IndicTrans2 is specialized for Indic languages and small enough for the budget. Translating sentence by sentence preserves exact alignment with the Hindi sentences and the groups of five.

**8. Stage memory, fail loudly, report honestly.** Load one model, use it, free it. If a model or adapter is missing, raise a clear error; never silently switch to a fallback. Report what was actually run and distinguish it from what is merely documented.

</div>

## A.3 The approach at a glance

| Stage | Input | Chosen model | Chosen decoder | Why this choice |
| --- | --- | --- | --- | --- |
| 1. Punctuation | flat Hindi word list | mBERT + LoRA token classifier (5 labels) | argmax over triangular-weighted window probabilities | one decision per word; preserves words; cheap; strong sentence-ending cues |
| 2. Ordering | list of plain Hindi sentences | mBERT + LoRA pair classifier (2 labels) | Borda over the full pairwise probability matrix | pairwise model generalizes to any length; Borda averages out local errors |
| 3. Translation | ordered Hindi sentences | IndicTrans2 200M encoder-decoder | beam search (4 beams) | Indic-specialized, fits budget, sentence alignment is exact |
| Orchestration | seven raw files | none | load/free one stage at a time | keeps the largest resident model under 1.5 GB |

The data flow is:

```text
raw tokens -> [stage 1: label each word] -> sentences
           -> [stage 2: score all pairs + Borda] -> ordered sentences
           -> [stage 3: translate each sentence] -> aligned English
           -> group every five sentences
```

## A.4 What we explicitly rejected

| Rejected approach | Why it was rejected |
| --- | --- |
| **Use an LLM to punctuate, reorder, and translate** | Forbidden by the assignment; hides the pipeline; violates the memory and reproducibility spirit. |
| **Full fine-tuning of mBERT** | Roughly 178M parameters of gradients and AdamW state; unnecessary under the 1.5 GB budget; higher overfitting risk on 25k windows. |
| **Sequence-to-sequence punctuation** | Could rewrite words, is slower, and makes word-level evaluation harder. Token classification is the natural formulation. |
| **A single end-to-end model for all three stages** | No aligned training data; impossible to evaluate per stage; too large for the budget. |
| **A direct permutation classifier for ordering** | Output size varies with the number of sentences; hard to train; pairwise decisions generalize better. |
| **Beam-only ordering** | Locally greedy; measured worse than Borda (mean tau 0.0921 vs 0.2007). |
| **TF-IDF-only ordering** | Ignores narrative cues; measured tau 0.0631. |
| **Document-level translation** | Breaks sentence alignment and the required groups of five; can exceed the model context. |
| **A fallback translation model** | Hides failures and makes results incomparable; the project raises an error instead. |
| **Keep all three models resident** | Sum of the stages exceeds the budget; staging is simpler and safer than quantization. |
| **Synthetic or LLM-generated training data** | Forbidden and unnecessary; Wikipedia already contains the supervision we need. |

## A.5 How the approach satisfies the assignment constraints

| Constraint | How the chosen approach satisfies it | Evidence |
| --- | --- | --- |
| No LLM | Only mBERT and IndicTrans2 are used; the evaluation report sets `"no_llm": true`. | `src/constants.py:22-24`, `reports/evaluation.json` |
| 1.5 GB GPU memory | LoRA, fp16 on CUDA, gradient checkpointing, one stage at a time. | `src/utils.py:74-99`, `src/pipeline.py:74-154` |
| Restore punctuation and boundaries | Five-label token classification with boundary reconstruction. | `src/punctuation/*`, boundary F1 0.8842 |
| Recover sentence order | Pairwise classifier plus Borda/clusters/beam decoders. | `src/ordering/*`, Borda tau 0.2007 |
| Translate Hindi to English | IndicTrans2 200M, sentence-level, aligned groups. | `src/translation/indictrans.py`, BLEU 34.125 |
| Produce groups of five | `group_lines(..., 5)` applied to Hindi and English. | `src/utils.py:292-294`, `src/pipeline.py:115,146-149` |
| Reproducibility | Pinned dependencies, seeded splits, JSON reports, executed notebook. | `requirements-colab.txt`, `reports/`, `notebooks/pipeline_demo.ipynb` |

## A.6 Why this approach is defensible in a viva

- **Every design choice has a stated alternative.** The table above is the short version; Chapter 28 expands it.
- **Every stage has its own metric.** You can defend punctuation without defending ordering, and ordering without defending translation.
- **The weaknesses are measured, not hidden.** Comma F1, question/exclamation F1, and ordering tau are reported honestly.
- **The memory claim is architectural, not wishful.** The code physically releases each model before loading the next.
- **The data story is clean.** Natural Wikipedia supervision, article-level splits, and no synthetic labels.

<div class='callout'>
<strong>If you remember one thing:</strong> the project succeeds because it chooses the <em>right decomposition</em> for the constraints. mBERT labels words, a pairwise model ranks sentences, a decoder turns rankings into an order, and IndicTrans2 translates. Each stage is small, measurable, and replaceable.
</div>

<div class='pagebreak'></div>

# Part I — NLP foundations from zero

## 5. What is natural language processing?

Natural language processing is the study of how to make computers do useful things with human language. “Useful things” includes classification (is this review positive?), extraction (who did what to whom?), generation (translate this sentence), retrieval (find the passage that answers this question), and structuring (turn a flat stream of words into sentences and paragraphs).

Language is hard for computers for reasons that are obvious to humans and invisible to code:

- **Ambiguity.** “I saw the man with the telescope” has two meanings. `बैंक` can mean a river bank or a financial bank.
- **Context dependence.** The meaning of a word depends on the sentence, the paragraph, and the world.
- **Compositionality.** The meaning of a sentence is built from its parts, but not by simple addition.
- **Variation.** Dialects, spelling variants, code-mixing, and noise.
- **Long-range structure.** A pronoun on page two may refer to a name on page one.

Classical NLP decomposed the problem into levels:

| Level | Studies | Example |
| --- | --- | --- |
| Phonology | Sounds | pronunciation, speech recognition |
| Morphology | Word structure | Hindi verb conjugation, plurals |
| Syntax | Sentence structure | subject-verb-object order |
| Semantics | Meaning | entailment, word sense |
| Pragmatics | Meaning in context | sarcasm, implicature |
| Discourse | Structure beyond a sentence | pronoun resolution, paragraph order |

This project touches morphology (Hindi is agglutinative and has rich inflection), syntax (sentence boundaries depend on clause structure), semantics (ordering depends on meaning and discourse), and generation (translation). It does not attempt speech or dialogue.

### 5.1 Rule-based, statistical, neural, and LLM approaches

The history of NLP is a sequence of increasingly flexible representations:

1. **Rule-based.** Hand-written grammars and dictionaries. Predictable, explainable, brittle. The legacy `baseline/` folder is in this tradition: dynamic-programming segmentation using verb endings, sentence starters, `NULL` hints, and a length prior.
2. **Statistical.** Counts and probabilities learned from data: n-gram language models, hidden Markov models, TF-IDF plus logistic regression. The `lessons/train.py` exercise is a tiny example of this family.
3. **Neural.** Dense vectors and gradient-based learning: recurrent networks, convolutional networks, and then transformers. The project's two mBERT stages are here.
4. **Large language models.** Very large transformers trained on broad text, often used through prompting or instruction tuning. The assignment explicitly forbids this family, so the project does not use one.

> **Why forbid LLMs?** The assignment is designed to test whether you can build a *pipeline* from task-specific components: data preparation, token classification, pairwise ranking, and translation. Asking a chat model to “fix this text” would hide all of that. It would also make the memory and reproducibility claims meaningless.

### 5.2 Where this project sits

| Assignment requirement | Technique used | Why |
| --- | --- | --- |
| Restore punctuation and boundaries | token classification with mBERT + LoRA | one decision per word; preserves the original words; small and fast |
| Recover sentence order | pairwise classification plus a decoder | order is a global structure, but a pairwise model generalizes to any number of sentences |
| Translate | IndicTrans2 encoder-decoder | dedicated Hindi-to-English model, much smaller than an LLM, strong on Indic languages |
| Fit 1.5 GB | fp16 on CUDA, LoRA, one stage at a time | keeps the largest resident model under the cap |

> **Check yourself.** Why is punctuation restoration formulated as token classification rather than as translation? *Because we do not want to rewrite the words; we want to attach one of five labels to each existing word. Translation-style generation would risk changing the text and would be far more expensive.*

## 6. From text to tokens

### 6.1 Text is Unicode, not bytes

A Python `str` is a sequence of Unicode code points. Each code point has a number. Hindi's Devanagari script occupies the range U+0900 to U+097F. Important characters in this project:

| Character | Code point | Name | Role |
| --- | --- | --- | --- |
| `।` | U+0964 | Devanagari danda | sentence end |
| `॥` | U+0965 | Devanagari double danda | verse/section end; treated as sentence end here |
| `?` | U+003F | question mark | question |
| `!` | U+0021 | exclamation mark | exclamation |
| `,` | U+002C | comma | comma |

Devanagari is not a simple alphabet. Consonants have an inherent vowel, and vowel signs (matras) are separate combining characters that attach to the consonant. For example, `अयोध्या` is six code points:

```text
अ  U+0905  DEVANAGARI LETTER A
य  U+092F  DEVANAGARI LETTER YA
ो  U+094B  DEVANAGARI VOWEL SIGN O      (combining)
ध  U+0927  DEVANAGARI LETTER DHA
्  U+094D  DEVANAGARI SIGN VIRAMA       (combining)
य  U+092F  DEVANAGARI LETTER YA
ा  U+093E  DEVANAGARI VOWEL SIGN AA      (combining)
```

The visual word is one grapheme cluster, but the string has multiple code points. This matters for any regex or tokenizer that assumes “one code point = one letter”.

> **Common mistake.** Using `re.findall(r'\w+', text)` on Hindi. In Python, `\w` matches letters and digits but not all combining marks, so matras are silently dropped:

```python
>>> import re
>>> re.findall(r'\w+', 'अयोध्या')
['अय', 'ध', 'य']
>>> re.findall(r'\w+', 'महालक्ष्मी')
['मह', 'लक', 'ष', 'म']
```

This is exactly why `src/data_prep.py` uses `str.split()` and `src/utils.py` implements `strip_punctuation()` without a word regex. Splitting on whitespace preserves every code point inside a token.

### 6.2 Word tokenization

Word tokenization splits a string into words. In English, whitespace plus punctuation rules are enough for many purposes; NLTK's `word_tokenize` additionally separates clitics and punctuation. In Hindi, whitespace is a strong signal because Hindi orthography separates words with spaces, but punctuation can be glued to the final word (`हुआ।`), and compounds may use hyphens (`राज्य-प्रबन्धन`).

The project's tokenizer is deliberately minimal:

```python
raw_tokens = text.split()
words = [token for token in raw_tokens if token != 'NULL']
```

Then, during data preparation, `strip_punctuation()` separates a trailing mark from the word and records it as a label. This is a *reversible* tokenization: the punctuation is stored as a label rather than thrown away, and `restore_punctuation()` can put it back.

### 6.3 Sentence tokenization

Sentence splitting is harder. In English, a period can end a sentence or an abbreviation. In Hindi, the danda is a strong signal, but a danda can also appear in abbreviations or verse numbering. The project uses a regex lookbehind:

```python
ARTICLE_PATTERN = re.compile(r'(?<=[।॥?!])\s+')
```

This splits at whitespace *after* a sentence-final mark. It is simple and works well for Wikipedia. It does not handle every edge case, which is acceptable because this is training-data extraction, not the final inference path. At inference time, sentence boundaries come from the trained classifier, not from this regex.

### 6.4 Subword tokenization

Modern transformer models rarely use whole words. A vocabulary of all Hindi words would be enormous and would still miss new words. Instead, models use **subword tokenization**: frequent words stay whole, rare words are split into smaller pieces.

The three main algorithms:

- **BPE (Byte Pair Encoding).** Start with characters (or bytes). Repeatedly merge the most frequent adjacent pair into a new symbol. This builds a vocabulary of frequent pieces.
- **WordPiece.** Similar in spirit, but merges are chosen by a likelihood criterion rather than raw frequency. Continuation pieces are marked with `##`. This is what BERT and mBERT use.
- **Unigram / SentencePiece.** Start with many candidate pieces and prune the vocabulary probabilistically. SentencePiece treats the input as a raw character stream, which is convenient for languages without spaces. IndicTrans2 uses a SentencePiece-style vocabulary.

A useful mental model: subword tokenization is a compression scheme learned from data. Common strings become single tokens; rare strings become several tokens. The model never sees an “unknown word” in the classical sense; it sees pieces.

#### A real mBERT example

The sentence `भारत एक विशाल देश है।` becomes:

```text
[CLS] भारत एक वि ##शा ##ल देश है । [SEP]
```

The word `विशाल` is split into `वि`, `##शा`, `##ल`; the `##` means “this piece continues the previous word”. Other examples:

| Word | mBERT pieces |
| --- | --- |
| `अयोध्या` | `अ`, `##य`, `##ोध`, `##्या` |
| `राज्य-प्रबन्धन` | `राज्य`, `-`, `प`, `##्र`, `##बन्ध`, `##न` |
| `महालक्ष्मी` | `म`, `##हा`, `##ल`, `##क्ष`, `##्मी` |
| `राम` | `राम` (one piece) |

> **Why this matters for labelling.** The punctuation label belongs to a *word*, but the model predicts a label for each *subword*. The project's dataset puts the word's label on the first subword and sets every continuation subword to `-100`, which the loss ignores. That way each word is supervised exactly once. See `src/punctuation/dataset.py:24-43`.

#### Special tokens, padding, truncation, attention masks

- `[CLS]` is a special classification token at the start. BERT's final hidden state for `[CLS]` is used as a sentence-level summary in sequence classification.
- `[SEP]` separates sequences. For a pair, the format is `[CLS] A [SEP] B [SEP]`.
- `[PAD]` fills shorter sequences in a batch so they have equal length.
- **Attention mask** is 1 for real tokens and 0 for padding, so the model ignores padding.
- **Truncation** cuts a sequence longer than the model's maximum length (128 for punctuation windows, 192 for sentence pairs).
- **`word_ids()`** maps each subword back to its original word index, with `None` for special tokens. This is the bridge between word-level labels and subword-level predictions.

```python
encoding = tokenizer(words, is_split_into_words=True, truncation=True,
                     max_length=128)
word_ids = encoding.word_ids()   # e.g. [None, 0, 0, 1, 1, 1, 2, None]
```

> **Check yourself.** Why can't we simply label every subword with the word's punctuation label? *Because then a five-subword word would contribute five identical training signals, and at inference time the same word would produce five possibly conflicting predictions. First-subword labelling gives one prediction per word and keeps the loss balanced.*
