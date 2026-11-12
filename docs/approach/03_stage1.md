# 6. Stage 1: punctuation as word-level token classification

<div class="decision">

**APPROACH CHOSEN.** Treat punctuation restoration as a five-class token-classification problem. Each existing word receives one label; the label is attached to the word's first subword during training; a class-weighted loss handles the imbalanced labels; overlapping windows with triangular weighting produce one prediction per word at inference.

</div>

## 6.1 Problem formulation

The input is a flat list of words $w_1,\ldots,w_n$ after `NULL` removal. The output is a label $y_i \in \{0,1,2,3,4\}$ for each word.

| Id | Label | Mark | Meaning |
| ---: | --- | --- | --- |
| 0 | `none` | nothing | word is inside a sentence |
| 1 | `comma` | `,` | a comma follows the word |
| 2 | `sentence_end` | `।` | the sentence ends after this word |
| 3 | `question` | `?` | a question ends after this word |
| 4 | `exclamation` | `!` | an exclamation ends after this word |

The label is reconstructed into text only after prediction:

```python
text = word + PUNCTUATION[label]
```

This is why the stage cannot hallucinate words. It never generates text; it chooses labels for existing tokens.

## 6.2 Why token classification instead of seq2seq

| Approach | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| **Token classification** | Exact word alignment; fast; small; deterministic; no hallucination; easy per-word evaluation | Cannot insert or rewrite words; rare labels hard; subword alignment needed | **Chosen** |
| Seq2seq punctuation | Can in principle rewrite or insert words | Slower; may change words; harder alignment; larger model | Rejected |
| LLM prompting | Fluent output | Forbidden; opaque; expensive; non-deterministic; not a pipeline | Rejected |
| Rule-based punctuation | No training | Brittle; does not scale; cannot use context well | Rejected as final stage |

The assignment cares about restoring punctuation without losing the input words. Token classification is the only formulation that guarantees that property.

## 6.3 Subword alignment

mBERT uses WordPiece. A Hindi word may become several subword tokens. The model predicts a label for every subword, but the task defines one label per word.

The dataset resolves this by:

1. tokenizing the word list with `is_split_into_words=True`,
2. reading `word_ids()` to map each subword to its word,
3. placing the word's label on the **first subword** of that word,
4. setting every continuation subword and special token to `-100`.

`CrossEntropyLoss(ignore_index=-100)` then ignores those positions.

<figure class="diagram">
<img src="approach_assets/04_punctuation.svg" alt="Punctuation architecture and subword alignment">
<figcaption><strong>Figure 6.1.</strong> The mBERT backbone sees subwords; the classifier predicts five logits per subword; training keeps only the first subword of each word; reconstruction inserts the predicted mark after the word.</figcaption>
</figure>

Why first subword?

- It gives exactly one supervision signal per word.
- It avoids averaging conflicting predictions across subwords.
- It is simple and stable across tokenizers.

The cost is that evidence in later subwords is ignored at the label position. The model still sees those subwords as context; only their loss positions are masked.

## 6.4 Model architecture

| Component | Choice |
| --- | --- |
| Backbone | `bert-base-multilingual-cased` |
| Backbone parameters | about 177.9M |
| Head | linear classifier over 5 labels |
| Adaptation | LoRA on query and value projections |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| Modules saved | classifier |
| Bias | none |
| Trainable parameters | 593,669 |
| Total parameters | 177,860,362 |
| Trainable share | 0.3338% |

The backbone is frozen. Only the low-rank matrices and the classifier are trained. The adapter directory contains the LoRA weights, the classifier, and the tokenizer.

## 6.5 Class imbalance and loss

The training distribution is dominated by `none`:

| Label | Count | Share |
| --- | ---: | ---: |
| `none` | 1,653,107 | 90.947% |
| `comma` | 59,324 | 3.264% |
| `sentence_end` | 104,840 | 5.768% |
| `question` | 236 | 0.013% |
| `exclamation` | 153 | 0.008% |

The class weight for label $c$ is:

$$
w_c = \min\left(20,\ \frac{N}{K \cdot N_c}\right)
$$

where $N$ is the total number of labelled positions, $K=5$, and $N_c$ is the count of label $c$. The cap at 20 prevents the two rarest classes from dominating the gradient.

The measured weights are:

| Label | Weight |
| --- | ---: |
| `none` | 0.2199 |
| `comma` | 6.1279 |
| `sentence_end` | 3.4675 |
| `question` | 20.0000 |
| `exclamation` | 20.0000 |

The loss is weighted cross entropy with `ignore_index=-100`. The model therefore sees far more `none` examples but each rare example contributes more.

<div class="warning">

**What class weights do not fix.** There are only 236 question and 153 exclamation examples in the entire training set. A weight of 20 makes each example count more, but it cannot invent diversity. The measured F1 for both classes is 0.0. This is a data limitation, not a tuning failure.

</div>

## 6.6 Training recipe

### 6.6.1 Default full-data recipe

| Setting | Value |
| --- | --- |
| Epochs | 3 |
| Train batch size | 8 |
| Eval batch size | 16 |
| Gradient accumulation | 2 |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Warmup | 6% |
| Best metric | validation boundary F1 |
| Checkpointing | one per epoch, keep best |
| fp16 | yes on CUDA |
| Gradient checkpointing | yes |
| Seed | 42 |

### 6.6.2 Measured local subset recipe

The checked-in adapter was trained with:

| Setting | Value |
| --- | --- |
| Epochs | 2 |
| Train batch size | 16 |
| Learning rate | 3e-4 |
| Train windows | 4,000 |
| Validation windows | 2,317 |
| Test windows | 2,208 |
| Trainable parameters | 593,669 |

The subset run is the source of the measured numbers in this document. The default three-epoch settings are the full-data recipe.

### 6.6.3 Why boundary F1 is the selection metric

Token accuracy is dominated by `none`. A model that predicts `none` everywhere can score about 0.91 accuracy while detecting no boundaries. Validation boundary F1 treats all three boundary classes as one positive class and measures whether the model actually finds them.

The metric is computed as:

$$
P = \frac{TP}{TP+FP},\qquad
R = \frac{TP}{TP+FN},\qquad
F_1 = \frac{2PR}{P+R}
$$

where a boundary prediction is positive if the label is `sentence_end`, `question`, or `exclamation`.

## 6.7 Inference

Inference is designed to use more context than a single 64-word window.

### 6.7.1 Overlapping windows

For a word list of length $n$:

- window size = 64 words,
- stride = 40 words,
- windows are clipped at the end,
- short inputs use one window.

<figure class="diagram">
<img src="approach_assets/05_windows.svg" alt="Overlapping windows and triangular weighting">
<figcaption><strong>Figure 6.2.</strong> A word may appear in several windows. Predictions near a window edge are weighted less than predictions near the centre.</figcaption>
</figure>

### 6.7.2 Triangular weighting

For a window of length $\ell$ and a word at position $i$ inside it:

$$
a_i = \max\left(0.05,\ 1 - \frac{|i - (\ell-1)/2|}{(\ell-1)/2 + 1}\right)
$$

The weight is 1 at the centre, falls toward the edges, and never goes below 0.05. A word that appears in several windows receives a weighted average of its class probabilities:

$$
\bar{p}_i = \frac{\sum_w a_{i,w}\,p_{i,w}}{\sum_w a_{i,w}}
$$

The final label is $\arg\max \bar{p}_i$.

Why triangular instead of uniform? A word at the edge of a window has less left or right context. Its prediction is less reliable. The triangular weight lets central predictions dominate while still using every observation.

### 6.7.3 Batching and precision

| Setting | Value |
| --- | --- |
| Inference window length | 64 words |
| Stride | 40 words |
| Maximum tokenizer length | 160 |
| Inference batch size | 16 |
| CUDA precision | fp16 |
| MPS precision | fp32 |

### 6.7.4 Reconstruction

After labels are predicted:

- `sentence_end` appends `।`,
- `comma` appends `,`,
- `question` appends `?`,
- `exclamation` appends `!`,
- `none` appends nothing.

A sentence is emitted whenever a boundary label occurs. If the file ends without a boundary, the remaining words are emitted as a final sentence. This is why the pipeline can never lose words: every word belongs to exactly one output sentence.

## 6.8 Measured results

`reports/evaluation.json` on 1,500 held-out windows and 108,898 words:

| Metric | Value |
| --- | ---: |
| Word accuracy | 0.8724 |
| Boundary precision | 0.8539 |
| Boundary recall | 0.9167 |
| Boundary F1 | 0.8842 |
| `none` F1 | 0.9285 |
| `comma` F1 | 0.3161 |
| `sentence_end` F1 | 0.8838 |
| `question` F1 | 0.0000 |
| `exclamation` F1 | 0.0000 |

The training summary's test split is slightly different because it uses the Trainer's test evaluation on the full test JSONL: accuracy 0.8688, boundary F1 0.8633. The evaluation report uses the first 1,500 windows and the same inference path as the pipeline, so it is the number to quote for deployed behaviour.

## 6.9 Pros of this stage design

1. **Word preservation.** No word can be inserted, deleted, or rewritten.
2. **Exact evaluation.** Every word has one gold label and one prediction.
3. **Efficiency.** A 178M encoder with LoRA is small enough for the budget.
4. **Deterministic inference.** The same input and adapter produce the same labels.
5. **Simple reconstruction.** Boundaries are a label, not a generation problem.
6. **Strong sentence endings.** The danda and finite-verb cues make `sentence_end` learnable.
7. **Cheap inference.** Punctuation took 2.31 seconds for all seven files on MPS.

## 6.10 Cons and failure modes

| Weakness | Cause | Evidence / effect |
| --- | --- | --- |
| **Comma F1 is low** | Comma placement is style-dependent and often optional. | 0.3161 |
| **Question/exclamation F1 is zero** | 236 / 153 training examples. | 0.0 / 0.0 |
| **Dialogue over-segmentation** | Wikipedia sentences are at least five words; short utterances look like endings. | sample-6 has 80 sentences from 507 words; sample-7 has 99 from 846. |
| **No word rewriting** | Boundary marks only; no missing word repair. | A malformed input token stays malformed. |
| **Subword evidence loss** | Only the first subword carries the label. | Later-subword cues are context, not supervised positions. |
| **Truncation risk** | A very long window can exceed 160 subwords. | Rare for 64-word windows, but possible with long compounds. |
| **Domain shift** | Trained on encyclopedic prose. | News and dialogue differ. |

## 6.11 Implementation map

| Function | File | Role |
| --- | --- | --- |
| `PunctuationDataset` | `src/punctuation/dataset.py` | word list to subword tensors and first-subword labels |
| `class_weights` | `src/punctuation/train.py` | inverse-frequency weights, capped at 20 |
| `WeightedTrainer` | `src/punctuation/train.py` | weighted cross entropy with `ignore_index=-100` |
| `compute_metrics` | `src/punctuation/train.py` | accuracy, boundary P/R/F1, per-class F1 |
| `PunctuationRestorer` | `src/punctuation/predict.py` | adapter loading, sliding windows, triangular aggregation |
| `restore` | `src/punctuation/predict.py` | labels to punctuated and plain sentences |
| `sliding_windows`, `window_weight` | `src/utils.py` | window layout and triangular weights |
| `strip_punctuation`, `restore_punctuation` | `src/utils.py` | reversible mark handling |

<div class="pagebreak"></div>
