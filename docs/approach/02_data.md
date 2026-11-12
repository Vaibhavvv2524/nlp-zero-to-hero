# 5. Data approach: create supervision by hiding information

<div class="decision">

**APPROACH CHOSEN.** Build the training data from Hindi Wikipedia by removing information that is already present. Punctuation is stripped and stored as labels; sentence order is shuffled inside windows; consecutive sentences become positive pair examples and their reversals become negatives. Use one article-level split for both learned stages, cap the data for tractability, and never synthesise labels with an LLM.

</div>

The data strategy is the foundation of the whole approach. The models are small and the supervision is natural; neither would be enough without a careful construction step.

## 5.1 Why Wikipedia

The assignment provides raw input files but no training labels. Wikipedia supplies three things that match the task:

| Signal | How it becomes supervision |
| --- | --- |
| Punctuation already present | Remove the mark from the final token and store it as a word-level label. |
| Sentences already ordered | Consecutive sentences give correct order; swapping gives incorrect order. |
| Large Hindi prose corpus | Enough articles to create 25,000 windows and 40,000 pairs without synthetic generation. |

This is **self-supervision by hiding**, not weakly supervised guessing. The label is exact: the mark that was removed is the mark the model must restore.

The dataset is `wikimedia/wikipedia`, configuration `20231101.hi`, streamed so that only the requested number of articles is kept. The default is 3,000 articles; the local run uses that count.

## 5.2 The data pipeline

<figure class="diagram">
<img src="approach_assets/03_data.svg" alt="Data construction pipeline">
<figcaption><strong>Figure 5.1.</strong> Wikipedia is parsed into sentences, split by article, and then turned into punctuation windows, ordering pairs, and held-out ordering articles. All three products share the same article split.</figcaption>
</figure>

### 5.2.1 Parse an article

`parse_article` performs the following steps:

1. Remove citation-like bracketed numbers with `re.sub(r"\[\d+\]", "", text)`.
2. Collapse whitespace.
3. Split on whitespace that follows a sentence-final mark using `(?<=[।॥?!])\s+`.
4. For each token, strip edge quotes and a trailing punctuation mark.
5. Keep tokens that contain Devanagari or at least one digit; drop the rest.
6. Rebuild sentence blocks at boundary labels.
7. Keep only sentences with 5–40 words.
8. Keep the article only if it produces at least three valid sentences.

The regex is deliberately narrow. It splits after `।`, `॥`, `?`, and `!`, not after every period, because Hindi prose usually uses the danda. The label extractor does recognise a final `.` as `sentence_end` when the token is not numeric, so period-final text contributes labels even if the article splitter does not cut there.

### 5.2.2 Tokenization: whitespace, not `\w`

Hindi words must be preserved exactly. A regex such as `\w+` drops Devanagari combining marks (matras) and breaks words. The project therefore uses `str.split()` and a small punctuation stripper that understands the marks in the corpus.

```python
raw_tokens = text.split()
words = [token for token in raw_tokens if token != "NULL"]
```

`strip_punctuation` then separates a trailing mark and returns `(word, label)`. This is reversible: `restore_punctuation` puts the predicted mark back.

### 5.2.3 Article-level splits

The split is by article id, not by window or sentence:

```text
shuffle article ids with seed 42
test  = last 5%   -> 150 articles
val   = previous 5% -> 150 articles
train = first 90% -> 2,700 articles
```

The same split is used for punctuation and ordering. No article can appear in two splits, so a window from a test article cannot share a document with a training window. This is the strongest practical guard against leakage.

## 5.3 Punctuation windows

Punctuation is learned from windows of whole sentences:

- Target window size: 64 words.
- Minimum window: 32 words.
- Maximum window: 96 words.
- Stride: 40 words.
- A window must contain at least two sentence blocks.
- Sentences inside the window are shuffled with the split's seeded RNG.
- The training set is capped at 25,000 windows.

### 5.3.1 Window construction algorithm

```text
index = 0
while index < len(sentences):
    add whole sentences until total >= 64 or adding another would exceed 96
    if 32 <= total <= 96 and at least 2 sentences:
        shuffle the sentence blocks
        concatenate words and labels
        emit one example
    advance by approximately 40 words, at least one sentence
```

Shuffling is not cosmetic. The assignment input has shuffled sentences. If windows kept natural order, the model could learn cross-sentence continuity that is absent at inference. Shuffling forces the model to rely on local word context.

### 5.3.2 Why windows overlap

The stride (40) is smaller than the window (64), so neighbouring windows overlap by about 24 words. Overlap gives the inference smoother coverage: a word near the edge of one window may be central in another. The training data also sees more contexts per article without creating independent documents.

## 5.4 Ordering pairs

Ordering supervision is built only from **consecutive** sentences. For each sampled position $i$:

- positive: `(sentence_i, sentence_{i+1}, 1)`
- negative: `(sentence_{i+1}, sentence_i, 0)`

Each natural pair contributes two examples, so the labels are exactly balanced. The default cap is 30 sampled positions per article; the positions are spread evenly across the article rather than randomly chosen, so long articles are represented across their full span.

### 5.4.1 Pair construction algorithm

```text
positions = 0 .. len(sentences)-2
if len(positions) > max_pairs:
    sample evenly across the range
for each sampled index:
    emit (A, B, 1)
    emit (B, A, 0)
```

Why both directions?

- **No position shortcut.** A model trained only on `(A,B,1)` could learn "the second sentence is the answer" or exploit ordering artifacts.
- **Balanced labels.** The classifier cannot achieve high accuracy by always predicting one class.
- **Symmetry at inference.** The score matrix uses $M_{ji}=1-M_{ij}$; balanced training makes that complement meaningful.

### 5.4.2 Why not random sentence pairs?

Random pairs would create many obviously unrelated pairs, making the task too easy and teaching the model to rely on topic similarity rather than discourse order. Consecutive pairs are hard: both sentences are from the same article, so the model must use coherence cues, not just subject matter.

## 5.5 Ordering evaluation articles

Kendall's tau needs complete articles, not isolated pairs. The test split contributes up to 40 articles with 8–70 sentences. Each article is stored with its true sentence order.

At evaluation time, an article is shuffled with a seeded RNG, scored, reordered, and compared with the truth. The same pairwise matrix is reused for Borda, clusters, beam, and the TF-IDF ablation, so the decoder comparison is fair.

## 5.6 Measured data statistics

The local run produced the following counts:

| Split | Articles | Punctuation windows | Ordering pairs |
| --- | ---: | ---: | ---: |
| train | 2,700 | 25,000 | 40,000 |
| validation | 150 | 2,317 | 3,000 |
| test | 150 | 2,208 | 3,000 |
| ordering evaluation | 40 held-out articles | — | — |

### 5.6.1 Label distribution

Training labels from `reports/data_prep_summary.json`:

| Label | Count | Share |
| --- | ---: | ---: |
| `none` | 1,653,107 | 90.947% |
| `comma` | 59,324 | 3.264% |
| `sentence_end` | 104,840 | 5.768% |
| `question` | 236 | 0.013% |
| `exclamation` | 153 | 0.008% |

The imbalance is the single most important fact about stage 1. A model that predicts `none` everywhere would already be about 91% accurate. That is why accuracy is not the headline metric and why class weights are mandatory.

## 5.7 What the data deliberately does not contain

| Missing item | Why it is absent | Consequence |
| --- | --- | --- |
| LLM-generated labels | The assignment forbids an LLM solution; synthetic labels would hide the pipeline. | Natural supervision only. |
| Aligned triplets | Wikipedia does not provide them. | The task is decomposed instead of trained end to end. |
| Dialogue and poetry | Wikipedia is encyclopedic prose. | The model over-segments short utterances. |
| Balanced question/exclamation marks | Questions and exclamations are rare in Wikipedia. | Their F1 is effectively zero. |
| Document-level English translations of the assignment files | Not available. | Translation is evaluated on IITB pairs and inspected qualitatively on pipeline output. |

## 5.8 Data validation

The verifier checks the data before any model is evaluated:

- exactly 3,000 articles are kept,
- split sizes are 2,700 / 150 / 150,
- punctuation train windows equal 25,000,
- ordering train pairs equal 40,000,
- every JSONL row count matches the summary,
- words and labels have equal length,
- every window has 32–96 words,
- every label is one of the five allowed labels,
- ordering pair labels are 0 or 1 and approximately balanced,
- no article id appears in two splits,
- ordering evaluation articles have 8–70 sentences.

The data checks are intentionally independent of the models. A data bug should fail before GPU time is spent.

## 5.9 Data risks and mitigations

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Wikipedia text drifts across snapshots | Counts and label distribution can change. | Pin `20231101.hi` and record the dataset/config. |
| Parser merges period-final sentences | Some long sentence blocks. | Labels still mark `.` boundaries; windowing remains valid. |
| Article split is random but finite | Sampling variance, especially 150 test articles. | Fixed seed, report split sizes, avoid overclaiming. |
| Subset adapters | Shipped models see 4,000 examples, not 25k/40k. | Separate subset and full-data claims. |
| Domain mismatch | Dialogue samples over-segment. | State the limitation; report per-sample counts. |
| Rare classes | Question/exclamation F1 zero. | Report honestly; do not hide behind accuracy. |

<div class="pagebreak"></div>
