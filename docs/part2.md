## 7. Embeddings and vector representations

A model cannot do arithmetic on the string `भारत`. It needs numbers. An **embedding** is a dense vector that represents a token, word, sentence, or document.

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> We do not train embeddings from scratch. We reuse mBERT's contextual embeddings, and we use TF-IDF only where a cheap lexical representation is enough (the ordering ablation and the clustering decoder). This keeps the learned representation strong and the extra machinery minimal.
</div>

### 7.1 One-hot and bag-of-words

The simplest representation is a vector with one dimension per vocabulary item. For a 100,000-word vocabulary, `भारत` might be a vector of 99,999 zeros and one one. This is interpretable but enormous, sparse, and semantically empty: `भारत` and `भारतीय` are as far apart as `भारत` and `banana`.

A **bag of words** counts word occurrences in a document and ignores order. It is the basis of TF-IDF.

### 7.2 TF-IDF

TF-IDF weights a term by how often it appears in a document, discounted by how common it is across all documents:

```text
tf-idf(t, d) = tf(t, d) * log(N / df(t))
```

- `tf(t,d)` is the frequency of term `t` in document `d`.
- `df(t)` is the number of documents containing `t`.
- `N` is the number of documents.

A term that appears everywhere (like “the”) gets a low weight; a term that is rare but appears in this document gets a high weight. The project uses TF-IDF in two places:

- **Ordering ablation:** a TF-IDF similarity matrix plus a greedy order, with no mBERT. This is the “can simple lexical similarity do the job?” baseline.
- **Clustering decoder:** character n-gram TF-IDF (`analyzer='char_wb'`, 2- to 4-grams) plus KMeans to group topically similar sentences before ordering.

Character n-grams are useful for Hindi because they capture morphology and tolerate spelling variation better than whole-word features. `char_wb` means n-grams are built within word boundaries, with padding at the edges.

### 7.3 Static word embeddings

Word2vec, GloVe, and fastText learn a vector per word from co-occurrence statistics:

- **Word2vec (skip-gram).** Predict context words from a target word. The learned input vectors become the embeddings. The objective for a word `w_t` and context window `c` is to maximize the sum of `log p(w_{t+j} | w_t)` over nearby context positions.
- **GloVe.** Factorizes a global co-occurrence matrix so that vector differences encode meaning relations.
- **fastText.** Represents a word as the sum of its character n-gram vectors, so it can handle unseen words and morphology.

These embeddings are **static**: the vector for `बैंक` is the same in every sentence. That is a fundamental limitation.

### 7.4 Contextual embeddings

Transformers produce **contextual** representations. The vector for a token is computed from the token plus its entire context, layer by layer. In BERT, the final hidden state for each token is a 768-dimensional vector that depends on the sentence. The same word in two different sentences can have different vectors.

For a sentence-level task, BERT's `[CLS]` vector is a learned summary of the whole sequence. For a token-level task, each token's final hidden state is passed through a classification head.

<div class='decision'>
<strong>Choice:</strong> mBERT contextual embeddings for both learned stages, not static word vectors. <strong>Reason:</strong> punctuation depends on syntax and local context; ordering depends on the relationship between two sentences. Static vectors cannot represent either. <strong>Cost:</strong> a 178M-parameter backbone, which we fit with LoRA and fp16.
</div>

### 7.5 Cosine similarity

To compare two vectors, the standard measure is cosine similarity:

```text
cos(u, v) = (u . v) / (||u|| * ||v||)
```

It measures the angle between vectors and ignores magnitude. It is used in the clustering decoder to choose a medoid (the member closest to the cluster centroid) and in the TF-IDF ablation.

> **Check yourself.** Why does the project use contextual mBERT embeddings for ordering but TF-IDF for clustering? *TF-IDF is cheap and good enough to group topically similar sentences; the expensive mBERT pair model is reserved for the finer before/after decisions. The evaluation compares this hybrid against a TF-IDF-only ablation.*

## 8. Transformers and BERT

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Both learned stages use the same encoder-only backbone, mBERT, with a small task-specific head. Stage 1 uses token classification; stage 2 uses sequence classification. The translation stage is the only encoder-decoder. We deliberately avoid a single multi-task model because separate heads are easier to train, evaluate, and reason about.
</div>

### 8.1 The basic neural network

A neural network is a stack of layers that transform vectors. A linear layer computes `y = Wx + b`. A nonlinearity such as ReLU or GELU lets the network learn curved decision boundaries. A loss function measures how wrong the output is; backpropagation computes gradients; an optimizer updates the weights.

For classification, the model produces **logits** `z`, one per class. **Softmax** turns them into probabilities:

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

The **cross-entropy loss** for the correct class `y` is `-log(p_y)`. If the model is confident and correct, the loss is near zero. If it is confident and wrong, the loss is large. This is the loss used in both mBERT stages (with class weights in punctuation).

### 8.2 Attention

The central operation of a transformer is **scaled dot-product attention**:

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

- `Q` (query), `K` (key), and `V` (value) are linear projections of the input.
- `Q K^T` computes a score for every pair of positions: how much should position `i` attend to position `j`?
- Dividing by `sqrt(d_k)` keeps the scores from growing too large as the dimension grows.
- Softmax turns the scores into attention weights.
- Multiplying by `V` produces a weighted sum of value vectors.

**Multi-head attention** runs several attention operations in parallel with different projections, then concatenates and projects the results. Different heads can learn different relationships: syntax, coreference, local order, and so on.

A transformer block adds:

- **Residual connections:** `x + f(x)`, which make deep networks trainable.
- **Layer normalization:** stabilizes activations.
- **Feed-forward network:** a two-layer MLP applied to each position independently, usually with a hidden size 4 times the model dimension.
- **Dropout:** randomly zeroes activations during training to reduce overfitting.

### 8.3 Encoder, decoder, and sequence-to-sequence

- An **encoder** reads the whole input and produces contextual representations. BERT is an encoder.
- A **decoder** generates output one token at a time, attending to what it has already generated and, in a sequence-to-sequence model, to the encoder's output.
- A **sequence-to-sequence** model combines both. IndicTrans2 is an encoder-decoder: the encoder reads Hindi, the decoder generates English.

BERT uses **bidirectional** attention: every token can attend to every other token. This is great for understanding. It is not directly suited to generation, because generation must not see the future. GPT-style models use **causal** attention: each token can attend only to earlier tokens.

### 8.4 BERT and mBERT

BERT is pretrained with two objectives:

1. **Masked language modelling (MLM).** Randomly mask some tokens and predict them from context. This forces the model to learn syntax and semantics.
2. **Next sentence prediction (NSP).** Predict whether sentence B follows sentence A. NSP is less important in modern practice, but it is part of the original BERT.

`bert-base-multilingual-cased` is the multilingual version, trained on Wikipedia in over 100 languages. Its configuration:

| Property | Value |
| --- | --- |
| Vocabulary size | 119,547 |
| Hidden size | 768 |
| Layers | 12 |
| Attention heads | 12 |
| Feed-forward size | 3,072 |
| Maximum positions | 512 |
| Parameters | about 178 million |

It is **cased**, so it distinguishes uppercase and lowercase; this matters little for Hindi but is the standard checkpoint. It has no task-specific head until we add one.

### 8.5 Task heads

A pretrained encoder is a backbone. To use it, we attach a **head**:

- **Token classification:** a linear layer maps each token's 768-dimensional hidden state to 5 logits (one per punctuation label).
- **Sequence classification:** the `[CLS]` vector is mapped to 2 logits (wrong order / correct order).
- **Sequence-to-sequence:** the encoder-decoder generates token by token; no simple classification head.

The head is randomly initialized and trained from scratch. This is why the notebook prints “Some weights of BertForTokenClassification were not initialized from the model checkpoint” when loading the base model. That warning is expected: the backbone is pretrained, the head is new.

### 8.6 Why mBERT for this project

- It is multilingual and has seen Hindi during pretraining.
- It is small enough (about 178M parameters) to fit the budget with LoRA and fp16.
- It is widely available, well documented, and works with the pinned `transformers` version.
- It supports both token classification and sequence classification with the same backbone.

Alternatives such as IndicBERT or MuRIL are also reasonable; mBERT was chosen for availability and because it is the model used in the course notes. The guide's design-decisions chapter discusses the trade-offs.

> **Check yourself.** Why does the ordering model use the `[CLS]` vector rather than averaging all token vectors? *The `[CLS]` representation is trained end-to-end for the pair classification objective, so it learns which aspects of the two sentences matter for their relative order. Averaging is a fixed, weaker pooling operation.*

## 9. Transfer learning, fine-tuning, and LoRA

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Freeze the mBERT backbone and train LoRA adapters on the query and value projections, plus the full classification head. This is the single most important memory decision in the project. It gives two task-specific models for roughly the cost of one backbone, and the adapters are small enough to save and reload.
</div>

### 9.1 Pretraining and fine-tuning

**Pretraining** teaches a model general language structure on a large corpus. **Fine-tuning** continues training on a smaller, task-specific dataset. The pretrained weights give the model a strong starting point, so fine-tuning needs much less data and time than training from scratch.

Full fine-tuning updates every weight. For a 178M-parameter model, that means storing gradients for all 178M parameters, storing optimizer state (AdamW keeps two moments per parameter), and accepting a large memory footprint and a risk of catastrophic forgetting.

The assignment's 1.5 GB budget makes full fine-tuning unattractive.

### 9.2 Parameter-efficient fine-tuning

**Parameter-efficient fine-tuning (PEFT)** freezes most of the pretrained model and trains a small number of new parameters. The most popular method is **LoRA (Low-Rank Adaptation)**.

### 9.3 The LoRA idea

A linear layer computes `y = Wx`. Full fine-tuning learns a new `W'`. LoRA instead freezes `W` and learns a low-rank update:

```text
W' = W + (alpha / r) * B A
```

where:

- `W` is the frozen pretrained matrix, shape `d_out x d_in`,
- `A` is a small matrix of shape `r x d_in`,
- `B` is a small matrix of shape `d_out x r`,
- `r` is the rank (16 in this project),
- `alpha` is a scaling constant (32 here).

The number of trainable parameters per adapted matrix is `r(d_in + d_out)`, which is tiny compared with `d_in * d_out`.

For mBERT, `d = 768` and `r = 16`:

```text
per matrix = 16 * (768 + 768) = 24,576
```

The project adapts the **query** and **value** projections in all 12 layers. That is 24 matrices:

```text
24 * 24,576 = 589,824
```

Add the fully trained classification head:

- punctuation: `768 * 5 + 5 = 3,845`
- ordering: `768 * 2 + 2 = 1,538`

So:

| Stage | LoRA params | Head params | Trainable total | Total model | Trainable fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Punctuation | 589,824 | 3,845 | 593,669 | 177,860,362 | 0.334% |
| Ordering | 589,824 | 1,538 | 591,362 | about 178.4M | 0.331% |

> **Why query and value?** Adapting `Q` and `V` changes what each token looks for and what it passes on, which is a strong lever for representation learning. Adapting the output projection is also common; adapting every matrix costs more. The choice is a standard, effective default.

### 9.4 LoRA hyperparameters

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `r` | 16 | rank of the update; higher = more capacity, more parameters |
| `lora_alpha` | 32 | scaling; effective scale is alpha/r = 2 |
| `lora_dropout` | 0.1 | dropout on the LoRA path |
| `target_modules` | `['query', 'value']` | which linear layers get adapters |
| `modules_to_save` | `['classifier']` | head trained and saved in full |
| `bias` | `'none'` | no bias training |

The adapter files are tiny. `adapter_model.safetensors` contains the LoRA matrices and the classifier head; the base model is loaded separately. At inference time, `PeftModel.from_pretrained(base_model, adapter_dir)` reconstructs the adapted model.

> **Important nuance.** LoRA does not reduce the memory needed to *store the base model*. It reduces the memory needed to *train*. At inference, the base weights are still there. The project's savings come from fp16 and from loading one model at a time, not from LoRA alone.

### 9.5 Adapter merging

LoRA updates can be merged into the base weights for deployment: `W' = W + (alpha/r) B A`. After merging, inference has no LoRA overhead. This project keeps the adapter separate for clarity and because the merged base would be large. IndicTrans2, by contrast, is used as a full model (no LoRA).

> **Check yourself.** If LoRA rank is increased from 16 to 64, what happens to the number of trainable parameters and to overfitting risk? *Parameters grow roughly linearly in `r` (4x), capacity grows, and overfitting risk grows, especially with only 4,000 training examples.*

## 10. Training mechanics

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Train each stage independently with a standard Hugging Face `Trainer`, a fixed seed, article-level splits, and a task-appropriate validation metric. Punctuation uses class weights and boundary F1; ordering uses balanced pairs and accuracy. Everything else is deliberately conventional so the unusual parts (LoRA, staging, Devanagari handling) stand out.
</div>

### 10.1 Data splits

The project splits **by article**, not by window or sentence:

- 2,700 articles for training,
- 150 for validation,
- 150 for test.

Splitting by article prevents leakage: two windows from the same article, or two sentence pairs from the same article, can be highly correlated. If one appeared in training and the other in test, the test score would be optimistic. The same article IDs define both the punctuation split and the ordering split, so the two stages are evaluated on the same held-out articles.

The punctuation split produces:

| Split | Articles | Windows | Words | `none` | `comma` | `sentence_end` | `question` | `exclamation` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2,700 | 25,000 | 1,817,660 | 90.95% | 3.26% | 5.77% | 0.013% | 0.008% |
| val | 150 | 2,317 | 168,661 | 91.38% | 2.99% | 5.60% | 0.009% | 0.021% |
| test | 150 | 2,208 | 159,105 | 90.90% | 3.20% | 5.88% | 0.018% | 0.014% |

The ordering split produces 40,000 training pairs, 3,000 validation pairs, and 3,000 test pairs, plus 40 held-out full articles for Kendall's tau.

### 10.2 Batch, step, epoch

- **Batch size** is the number of examples processed before one optimizer update.
- **Step** is one optimizer update.
- **Epoch** is one full pass over the training set.

With 25,000 windows, batch size 8, and gradient accumulation 2, one epoch has about `ceil(25,000 / (8 * 2)) = 1,563` optimizer steps. The effective batch size is 16 examples per update.

### 10.3 Learning rate, warmup, weight decay

- **Learning rate (LR)** controls step size. Too high diverges; too low learns slowly.
- **Warmup** gradually increases the LR from near zero to the target over the first 6% of steps, which stabilizes early training.
- **Weight decay** penalizes large weights (a regularizer). The project uses 0.01.
- **AdamW** is the optimizer: Adam with decoupled weight decay. It maintains a moving average of gradients and squared gradients per parameter.

### 10.4 Gradient accumulation and checkpointing

- **Gradient accumulation** sums gradients over several mini-batches before stepping. It simulates a larger batch without the memory cost.
- **Gradient checkpointing** stores fewer activations during the forward pass and recomputes them during the backward pass. It trades compute for memory. This is essential for fitting mBERT in a small budget.

### 10.5 Mixed precision

**fp16** stores activations and weights in 16-bit floats, roughly halving memory and speeding up math on supported GPUs. The project enables fp16 only on CUDA:

```python
fp16=torch.cuda.is_available()
```

On Apple MPS, training runs in fp32 because this torch build's fp16/MPS path is not reliable. Inference uses fp16 on CUDA and fp32 otherwise.

### 10.6 Seeds and reproducibility

`set_seed(42)` seeds Python, NumPy, and PyTorch. The data preparation uses seeded `random.Random` instances for shuffling and splitting. Exact reproducibility across hardware is never guaranteed (different GPUs, different kernels, nondeterministic operations), but the same machine and seed give stable results.

### 10.7 Class weights

The punctuation dataset is extremely imbalanced: 90.9% of words take no mark. A plain cross-entropy loss would learn to predict `none` almost always and still achieve high accuracy. The project uses inverse-frequency class weights:

```text
w_c = N / (K * n_c)
```

where `N` is the total number of labels, `K = 5` is the number of classes, and `n_c` is the count of class `c`. Weights are capped at 20 to avoid instability.

Using the training counts:

| Class | Count | Raw weight | Capped weight |
| --- | ---: | ---: | ---: |
| none | 1,653,107 | 0.220 | 0.220 |
| comma | 59,324 | 6.128 | 6.128 |
| sentence_end | 104,840 | 3.467 | 3.467 |
| question | 236 | 1,540.4 | 20.0 |
| exclamation | 153 | 2,375.3 | 20.0 |

The cap prevents the two rare classes from dominating the loss. It does not fully solve the rarity problem; see the results chapter.

`WeightedTrainer` overrides `compute_loss` to use `CrossEntropyLoss(weight=...)` with `ignore_index=-100`, so continuation subwords do not contribute.

### 10.8 Model selection

At the end of each epoch, the trainer evaluates on the validation set. It saves the checkpoint with the best validation metric:

- punctuation: best **boundary F1** (not accuracy, because accuracy is dominated by `none`),
- ordering: best **accuracy** (the classes are balanced, so accuracy is meaningful).

`load_best_model_at_end=True` restores the best checkpoint after training. `save_total_limit=1` keeps only the most recent checkpoint to save disk.

### 10.9 Exact training settings

| Setting | Punctuation | Ordering |
| --- | ---: | ---: |
| Base model | `bert-base-multilingual-cased` | `bert-base-multilingual-cased` |
| Task | token classification (5 labels) | sequence classification (2 labels) |
| LoRA `r` / `alpha` / dropout | 16 / 32 / 0.1 | 16 / 32 / 0.1 |
| LoRA targets | query, value | query, value |
| Head saved | classifier | classifier |
| Default epochs | 3 | 2 |
| Default batch size | 8 | 8 |
| Effective batch | 16 (accum 2) | 16 (accum 2) |
| Learning rate | 3e-4 | 2e-4 |
| Weight decay | 0.01 | 0.01 |
| Warmup | 6% | 6% |
| Max sequence length | 128 | 192 |
| fp16 | CUDA only | CUDA only |
| Gradient checkpointing | yes | yes |
| Best metric | boundary F1 | accuracy |
| Seed | 42 | 42 |

The locally verified adapters used different sample counts and epochs because MPS is slow:

| Setting | Punctuation (local) | Ordering (local) |
| --- | ---: | ---: |
| Training examples | 4,000 windows | 4,000 pairs |
| Epochs | 2 | 1 |
| Batch size | 16 | 16 |
| Learning rate | 3e-4 | 2e-4 |
| Test boundary F1 | 0.8633 | — |
| Test macro F1 | — | 0.6534 |
| Trainable parameters | 593,669 | 591,362 |
| Total parameters | 177,860,362 | about 178.4M |

> **Be precise in the viva.** The reported pipeline scores (boundary F1 0.8842, ordering tau 0.2007) come from these subset adapters, not from a full-data run. The full-data recipe is in the notebook and README; it has not been executed to completion in this repository. Saying otherwise would be a factual error.

### 10.10 Diagnosing training

Signals to watch:

- **Loss goes down, validation F1 goes up:** learning.
- **Training loss goes down, validation loss goes up:** overfitting; reduce epochs, increase dropout, add data.
- **Both losses flat:** learning rate too low, or labels misaligned.
- **Loss becomes NaN:** learning rate too high, fp16 instability, or bad class weights.
- **Accuracy high but F1 low:** class imbalance; look at per-class scores.
- **Boundary F1 high but comma F1 low:** the model is good at sentence endings but weak on commas; inspect comma examples.

The local training summaries show healthy learning: punctuation eval loss 0.4399, boundary F1 0.8246 after 2 epochs; ordering eval loss 0.5827, macro F1 0.6768 after 1 epoch. The ordering model is not overfit after one epoch, which is why the full recipe uses two.

> **Check yourself.** Why is boundary F1 a better model-selection metric than token accuracy for punctuation? *Because 90.9% of tokens are `none`; a model that predicts `none` everywhere gets 90.9% accuracy but finds zero boundaries. Boundary F1 forces the model to actually detect sentence endings.*

## 11. Inference and decoding

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Keep the inference path as close as possible to the training distribution, then add only the minimum decoding logic needed. Punctuation uses overlapping windows with triangular weighting; ordering uses an explicit global decoder; translation uses standard beam search. No stage relies on an LLM or a hidden heuristic.
</div>

### 11.1 Argmax and softmax

For a classification model, the simplest decoder is **argmax**: choose the class with the highest logit. Equivalently, choose the highest softmax probability. The punctuation model uses argmax after averaging probabilities across windows.

### 11.2 Sliding windows and triangular weighting

A document can be longer than the model's maximum input length. The punctuation model uses 64-word windows with stride 40. A word near the edge of a window has less context on one side, so its prediction is less reliable. The project averages predictions with a **triangular weight**:

```text
w(p, L) = max(0.05, 1 - |p - c| / (c + 1)),  c = (L - 1) / 2
```

where `p` is the position in the window and `L` is the window length. Words in the middle get weight near 1; words at the edges get weight near 0.05. Predictions are accumulated as `score[word] += weight * probability` and divided by the total weight.

Why this works: every word is seen in multiple windows (except near document edges), and the windows where it is central dominate. The result is smoother and more accurate than using a single window.

### 11.3 Batching

Inference processes windows in batches (16 for punctuation, 32 pairs for ordering, 4 sentences for translation). Batching improves GPU utilization. The code sorts nothing; it simply chunks the list.

### 11.4 Beam search

For sequence generation, greedy decoding picks the most likely next token at each step. **Beam search** keeps the `k` most likely partial sequences (the beam) and expands them, which can find a better global sequence. IndicTrans2 uses `num_beams=4`, `max_length=256`, and `early_stopping=True`.

Beam search is not used for punctuation or pair classification because those are single-step decisions, not sequences.

### 11.5 Ordering decoders

The ordering model produces a matrix `M` where `M[i,j] = P(sentence i before j)`. Three decoders turn this matrix into a permutation:

- **Borda (default):** score each sentence by its mean probability of coming before all others, then sort descending.
- **Beam:** build a chain, keeping the best partial chains, scoring each adjacent pair with `M`.
- **Clusters:** TF-IDF + KMeans to group sentences, order cluster medoids with the pair model, then beam-search inside each cluster.

The evaluation compares all three plus a TF-IDF-only ablation. Borda wins on mean Kendall's tau.

### 11.6 Complexity

For `n` sentences, the pair model evaluates `n(n-1)/2` pairs. For 99 sentences that is 4,851 pairs. This quadratic cost is why ordering is the slowest stage in the pipeline (130.53 seconds for all seven files on MPS). Clustering reduces the number of expensive pairs inside clusters, but the project still computes the full matrix for evaluation.

> **Check yourself.** Why is `M[j,i] = 1 - M[i,j]` valid? *The classifier is trained on both orders of every pair, so it approximates a symmetric preference. The code explicitly fills the transpose as the complement to keep the matrix consistent and avoid a second model call.*

## 12. Metrics: what the numbers mean

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Report the metric that matches the decision being made, not a single headline number. Boundary F1 for punctuation (not accuracy), pair F1 plus Kendall's tau for ordering (not pair accuracy alone), and BLEU plus chrF for translation (not one metric).
</div>

### 12.1 Confusion matrix, accuracy, precision, recall, F1

For binary decisions, define:

- **True positive (TP):** predicted positive, actually positive.
- **False positive (FP):** predicted positive, actually negative.
- **True negative (TN):** predicted negative, actually negative.
- **False negative (FN):** predicted negative, actually positive.

Then:

```text
accuracy  = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

Accuracy is intuitive but misleading under imbalance. Precision asks “when the model says positive, how often is it right?” Recall asks “of all real positives, how many did the model find?” F1 balances the two.

### 12.2 Macro, micro, weighted

- **Macro average:** compute the metric per class and average, giving each class equal weight. Good when rare classes matter.
- **Micro average:** pool all decisions and compute globally; dominated by frequent classes.
- **Weighted average:** average per-class metrics weighted by class support.

The ordering pair metrics use macro averaging because the two classes are balanced but we still want a symmetric view. The punctuation per-class F1 is reported class by class, which is more informative than any single average.

### 12.3 Boundary F1

The punctuation model predicts five labels. For the assignment, what matters most is whether a word is a **sentence boundary**. Define:

- boundary = label in `{sentence_end, question, exclamation}`.
- non-boundary = `{none, comma}`.

Then boundary precision/recall/F1 treat boundary detection as a binary problem. This collapses the rare question/exclamation classes into the boundary class, which is useful because those classes are too rare to evaluate separately. It still does not fix the fact that the model rarely predicts them.

### 12.4 Per-class F1 and the accuracy trap

The reported punctuation results:

| Metric | Value |
| --- | ---: |
| Token accuracy | 0.8724 |
| Boundary precision | 0.8539 |
| Boundary recall | 0.9167 |
| Boundary F1 | 0.8842 |
| F1 `none` | 0.9285 |
| F1 `comma` | 0.3161 |
| F1 `sentence_end` | 0.8838 |
| F1 `question` | 0.0 |
| F1 `exclamation` | 0.0 |

The accuracy of 87.24% sounds high, but a trivial “always `none`” model would get about 90.9% on this distribution. The meaningful number is boundary F1 0.8842: the model finds sentence endings well (sentence-end F1 0.8838) but is weak on commas (0.3161) and never predicts question or exclamation marks. This is a direct consequence of the training distribution.

> **Do not compare training-time and pipeline-time punctuation scores as if they were the same experiment.** The training summary's test boundary F1 (0.8633) is computed with a single truncated window per example, labels on first subwords only. The evaluation's 0.8842 uses sliding-window inference with triangular weighting on 1,500 test windows. Different inference procedures produce different numbers.

### 12.5 Pair accuracy and macro F1 for ordering

The ordering test set has 3,000 balanced pairs. Results:

| Metric | Value |
| --- | ---: |
| Pair accuracy | 0.6543 |
| Macro precision | 0.6560 |
| Macro recall | 0.6543 |
| Macro F1 | 0.6534 |

A random classifier gets 0.5. The model is clearly learning, but the margin is modest. Pairwise accuracy is not the same as recovering a full order: a model can be right on many pairs and still produce a poor global order because errors interact.

### 12.6 Kendall's tau

Kendall's tau measures rank correlation between the recovered order and the true order. For every pair of items, count whether their relative order agrees or disagrees:

```text
tau = (concordant - discordant) / (concordant + discordant)
```

- `tau = 1`: perfect order.
- `tau = 0`: no better than random.
- `tau = -1`: exactly reversed.

The evaluation shuffles each held-out article with a fixed seed, recovers an order, and computes tau against the original order. Results on 40 articles:

| Decoder | Mean tau | Median tau | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Borda | 0.2007 | 0.2304 | -0.1775 | 0.6444 |
| Clusters | 0.1220 | 0.1081 | -0.2652 | 0.7143 |
| Beam | 0.0921 | 0.0693 | -0.2467 | 0.7143 |
| TF-IDF ablation | 0.0631 | 0.0589 | -0.4847 | 0.5714 |

Borda is the best on average, which justifies making it the default. Some articles have negative tau, meaning the recovered order is worse than random for that article. This happens when the pairwise model is confidently wrong about a topic shift. Max tau around 0.64-0.71 shows that some articles are recovered reasonably well.

### 12.7 BLEU

BLEU measures n-gram overlap between a hypothesis translation and one or more references, with a brevity penalty:

```text
BLEU = BP * exp(sum_{n=1..N} w_n * log(p_n))
```

- `p_n` is the clipped n-gram precision: the fraction of hypothesis n-grams that appear in the reference, with counts clipped to the reference count.
- `BP = min(1, exp(1 - r/c))` where `r` is reference length and `c` is hypothesis length; it penalizes short outputs.
- Usually `N = 4` and weights are uniform.

BLEU rewards exact phrasing and is sensitive to word order. It correlates imperfectly with human judgment, especially for paraphrases.

### 12.8 chrF

chrF computes character n-gram precision and recall and combines them with an F-score. It is more tolerant of morphology and paraphrase than BLEU, which is useful for Hindi-English. The project uses `CHRF(word_order=2)`, which also considers word bigrams.

### 12.9 Translation results

On 100 IITB Hindi-English test sentences:

| Metric | Value |
| --- | ---: |
| BLEU | 34.125 |
| chrF | 59.686 |

These are respectable for a 200M-parameter model on a general benchmark. The sample translations show good meaning preservation with some paraphrase and occasional literal phrasing. BLEU and chrF are corpus-level scores; individual sentences can differ a lot.

> **Check yourself.** A model gets 95% token accuracy on punctuation but boundary F1 0.2. What is happening? *It is predicting the majority class almost everywhere. Accuracy is high because most words take no mark, but it misses most boundaries. This is exactly why the project reports boundary F1 and per-class F1.*

## 13. Data ethics, licensing, and responsible claims

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Use public, natural data; respect gated-model terms; keep all computation local; and state clearly what was and was not executed. The project's credibility rests on not overclaiming.
</div>

### 13.1 Data source

Training data comes from Hindi Wikipedia (`wikimedia/wikipedia`, config `20231101.hi`). Wikipedia text is publicly available under CC BY-SA and GFDL. The project uses it to learn punctuation and sentence order. It does not redistribute the raw dump; it stores derived datasets locally and ignores them in git.

The evaluation uses `cfilt/iitb-english-hindi` for BLEU/chrF. Check the dataset card for its license and terms before redistribution.

### 13.2 Model licenses and gating

- `bert-base-multilingual-cased` is publicly available on Hugging Face.
- `ai4bharat/indictrans2-indic-en-dist-200M` is **gated**: you must accept its terms on Hugging Face and authenticate before downloading. The local verification used an ungated merged mirror because the gated repository returned 401 without a token.
- Always check the model card for the license and intended use. A model's availability does not automatically mean unrestricted commercial use.

### 13.3 Privacy

The inputs are assignment texts, not personal user data. The pipeline does not log text to external services, does not call an API, and does not send data to an LLM. All computation is local or on the user's Colab runtime. This is a strong privacy property compared with a hosted chat model.

### 13.4 Bias and failure modes

Wikipedia is not a neutral sample of Hindi. It over-represents formal, encyclopedic text and under-represents dialogue, social media, and dialect. The model therefore:

- handles encyclopedic prose better than conversation,
- rarely predicts question/exclamation marks because Wikipedia uses them rarely,
- may impose a formal register on informal text.

Translation models can also reflect gender and cultural biases in their training data. The project does not attempt to measure or mitigate these biases; it is a course assignment, not a production system.

### 13.5 Honest reporting

The guide and README distinguish what was actually run locally (subset adapters, MPS), what is documented but not run (full-data Colab training), and what is skipped gracefully (translation evaluation if the dataset is unavailable). This honesty matters. A viva examiner will respect “we trained on 4,000 examples and here is the measured effect” far more than an inflated claim that cannot be reproduced.

> **Check yourself.** Why is “no LLM” both a technical and an ethical claim? *Technically, it means every output comes from a small, inspectable model. Ethically, it means no data leaves the machine and no opaque hosted service influences the result.*

## 14. The end-to-end mental model

Before diving into code, hold this picture in your head:

1. **Words in, labels out.** Stage 1 does not rewrite words. It attaches one of five labels to each word. The words are preserved exactly (after `NULL` removal).
2. **Labels define sentences.** A boundary label closes a sentence. The sentence text is the words plus their predicted marks.
3. **Sentences are nodes.** Stage 2 treats each sentence as a node and asks a pairwise question about every pair. The output is a matrix of probabilities, not an order.
4. **A decoder turns the matrix into an order.** This is a separate algorithmic step; the model does not directly output a permutation.
5. **Translation is sentence-level.** Each ordered sentence is translated independently. Grouping happens after translation, by position.
6. **Memory is managed by staging.** Load, use, free, repeat. The budget applies to the largest stage.
7. **Every stage has a metric.** Token/boundary F1 for punctuation, pair F1 and Kendall's tau for ordering, BLEU/chrF for translation.

If you can explain these seven points and the data shapes, you can explain the project.
