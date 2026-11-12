# 7. Stage 2: ordering with pairwise comparison and global decoding

<div class="decision">

**APPROACH CHOSEN.** Do not ask a model to generate a permutation. Train a binary classifier to answer one local question — does sentence A come before sentence B? — then combine many local answers into a global order with a Borda-like score. Cluster and beam decoders are implemented as alternatives; the measured winner is Borda.

</div>

## 7.1 Problem formulation

Given $n$ sentences in unknown order, produce a permutation $\pi$ such that $\pi(0),\ldots,\pi(n-1)$ is the original document order.

The learning target is not the permutation. It is the pairwise relation:

$$
f(A,B) = P(A \text{ precedes } B \mid A,B)
$$

A pair classifier is easier to train than a permutation generator because:

- the label is binary and exactly balanced,
- every consecutive sentence pair yields two examples,
- the same scorer can order documents of any length,
- the score matrix exposes uncertainty and supports several decoders.

The cost is the composition problem: local pairwise accuracy does not automatically produce a globally consistent order.

## 7.2 Why pairwise instead of listwise or generative ordering

| Approach | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| **Pairwise classifier + decoder** | Balanced labels; reusable for any $n$; interpretable score matrix; cheap per pair | $O(n^2)$ pairs; local errors can form cycles; decoder matters | **Chosen** |
| Listwise classifier | One call per document | Needs a fixed maximum length; hard to train; poor generalization | Rejected |
| Seq2seq permutation | Directly emits order | Can invent or drop sentences; slow; needs permutation training data | Rejected |
| LLM ordering | Fluent, zero-shot | Forbidden; expensive; non-deterministic; opaque | Rejected |
| TF-IDF similarity only | No training | Measures topic, not discourse order; weak baseline | Kept as ablation |

The assignment forbids an LLM solution and demands an inspectable pipeline. Pairwise scoring is the most inspectable learned ordering method.

## 7.3 Pair construction

Only consecutive sentences are used. For each position $i$:

```text
positive: (sentence_i, sentence_{i+1}) -> 1
negative: (sentence_{i+1}, sentence_i) -> 0
```

### 7.3.1 Sampling

The default is 30 positions per article, sampled evenly:

```text
positions = 0 .. len(sentences)-2
if len(positions) > max_pairs:
    step = len(positions) / max_pairs
    positions = [positions[int(i * step)] for i in range(max_pairs)]
```

Even sampling covers the start, middle, and end of long articles. Random sampling could miss entire sections.

### 7.3.2 Why reversals are necessary

Without negative reversals, the classifier could learn a trivial cue:

- "the first input is always earlier,"
- "the longer sentence is always earlier,"
- "the sentence with a discourse marker is always second."

Reversing the same pair removes those shortcuts: the only difference between the positive and negative example is the order of the two texts.

The local training set is 4,000 pairs (2,000 natural positions × 2 directions). The full-data default is 40,000.

## 7.4 Model architecture

<figure class="diagram">
<img src="approach_assets/06_ordering.svg" alt="Pairwise ordering architecture">
<figcaption><strong>Figure 7.1.</strong> Two sentences enter mBERT as a sentence pair. The classifier produces a binary probability. Every unordered pair is scored once; the matrix is completed by complementarity.</figcaption>
</figure>

| Component | Choice |
| --- | --- |
| Backbone | `bert-base-multilingual-cased` |
| Input format | `[CLS] A [SEP] B [SEP]` |
| Max sequence length | 192 |
| Classifier | 2 labels: `wrong_order`, `correct_order` |
| Adaptation | LoRA on query and value projections |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| Modules saved | classifier |
| Trainable parameters | 591,362 |
| Total parameters | 178,446,340 |
| Trainable share | 0.3314% |

The same mBERT checkpoint is used as in stage 1, but this is a separate adapter and a separate classification head. The two stages do not share trained weights at runtime.

## 7.5 Training recipe

### 7.5.1 Default full-data recipe

| Setting | Value |
| --- | --- |
| Epochs | 2 |
| Train batch size | 8 |
| Eval batch size | 16 |
| Gradient accumulation | 2 |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Warmup | 6% |
| Best metric | validation accuracy |
| fp16 | yes on CUDA |
| Gradient checkpointing | yes |
| Seed | 42 |

### 7.5.2 Measured local subset recipe

| Setting | Value |
| --- | --- |
| Epochs | 1 |
| Train batch size | 16 |
| Learning rate | 2e-4 |
| Train pairs | 4,000 |
| Validation pairs | 3,000 |
| Test pairs | 3,000 |
| Trainable parameters | 591,362 |

The checked-in ordering adapter is the subset run. Its validation accuracy is 0.6770 and its test accuracy is 0.6543.

### 7.5.3 Loss

The task is balanced binary classification. The default cross-entropy loss is sufficient; no class weighting is used. The label distribution is exactly 50/50 by construction.

## 7.6 From pairs to a matrix

For $n$ sentences, the scorer builds an $n \times n$ matrix $M$:

$$
M_{ij} = P(\text{sentence } i \text{ precedes sentence } j)
$$

Implementation details:

- The diagonal is fixed at 0.5: a sentence compared with itself is neutral.
- Only unordered pairs $(i,j)$ with $i<j$ are sent to the model.
- $M_{ij}$ receives the model probability.
- $M_{ji}=1-M_{ij}$ by complementarity.
- Scores are computed in batches of 32.
- The matrix is reused by every decoder.

For 99 sentences, this is $\binom{99}{2}=4{,}851$ forward passes, not $99^2$. Batched inference kept all seven assignment files at 130.53 seconds total on MPS.

A good matrix has high probabilities above the diagonal and low probabilities below it. The lower triangle should be the complement of the upper triangle. The measured matrix does not satisfy that property perfectly, which is exactly why decoding is a separate problem.

## 7.7 The four decoders

### 7.7.1 Borda-like mean score (default)

For each sentence $i$:

$$
s_i = \frac{1}{n-1}\sum_{j \ne i} M_{ij}
$$

Sort sentences by $s_i$ descending.

Interpretation: a sentence that is predicted to come before many other sentences has a high average "earliness" and is placed early. This is not literal Borda points from rank positions; it is a Borda-style aggregation of pairwise probabilities.

Why it works:

- every pair contributes to every score,
- one wrong pair cannot dominate a sentence's average,
- no hard cluster boundaries,
- no greedy chain that locks in an early mistake.

It is also the cheapest decoder: sorting $n$ floats after the matrix is built.

### 7.7.2 Beam search

Beam search builds a chain directly:

```text
starts = top beam_width sentences by mean before-score
beams = single-sentence paths
while any beam is incomplete:
    extend each beam with every remaining sentence
    new_score = old_score + M[last, next]
    keep the best beam_width paths
return best complete path
```

| Setting | Value |
| --- | --- |
| Beam width | 8 |
| Start candidates | 8 highest earliness scores |
| Path score | sum of adjacent pair probabilities |
| Complexity | roughly $O(n^2 \cdot b)$ scoring operations plus sorting |

Beam search optimises adjacency, not the whole matrix. It can be better when local transitions are strong, but a single bad transition early in the path is hard to undo.

### 7.7.3 TF-IDF clusters

The cluster decoder reduces the number of expensive mBERT comparisons by grouping sentences first:

1. Vectorise sentences with TF-IDF using character 2–4 grams, `sublinear_tf=True`.
2. Cluster with KMeans; default $k=\mathrm{round}(n/6)$, clamped to 1–12.
3. Choose the medoid of each cluster: the member with highest cosine similarity to the cluster centroid.
4. Order clusters greedily using the medoid submatrix and mean before-score.
5. Order each cluster internally with beam search.
6. Concatenate cluster orders.

If clustering produces one cluster, the decoder falls back to global beam search. If there are fewer than four sentences, it returns the identity order.

Why character n-grams? Hindi inflection and compounds make word-level overlap brittle. Character 2–4 grams capture shared morphology and spelling.

Why clusters at all? The notes suggest reducing pairwise model calls. In practice, clustering also introduces a hard assumption: topically similar sentences are contiguous. That assumption is often false in encyclopedic prose, which explains the measured result.

### 7.7.4 TF-IDF ablation

This decoder uses no mBERT at all:

1. Build the TF-IDF cosine similarity matrix.
2. Zero the diagonal.
3. Greedily pick the sentence with the highest mean similarity to the remaining sentences.
4. Repeatedly pick the most similar remaining sentence to the current one.

It measures how much of the ordering signal is just lexical overlap. It is a baseline, not a candidate for the final pipeline.

## 7.8 Measured results

### 7.8.1 Pair classification

| Split | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.6770 | 0.6786 | 0.6778 | 0.6768 |
| test | 0.6543 | 0.6560 | 0.6543 | 0.6534 |

The model is meaningfully above chance (0.50) but far from reliable.

### 7.8.2 Kendall's tau on 40 held-out articles

| Decoder | Mean τ | Median τ | Min τ | Max τ |
| --- | ---: | ---: | ---: | ---: |
| **Borda** | **0.2007** | 0.2304 | −0.1775 | 0.6444 |
| Clusters | 0.1220 | 0.1081 | −0.2652 | 0.7143 |
| Beam | 0.0921 | 0.0693 | −0.2467 | 0.7143 |
| TF-IDF ablation | 0.0631 | 0.0589 | −0.4847 | 0.5714 |

Borda is the clear winner, but the absolute score is modest. A mean τ of 0.20 means the recovered order is better than random but far from the true document order. This is the weakest stage of the pipeline and should be presented as such.

### 7.8.3 Why pair accuracy does not become order accuracy

A pair classifier answers isolated questions. A global order requires those answers to be mutually consistent. Small local errors can create cycles:

$$
A \prec B,\quad B \prec C,\quad C \prec A
$$

No permutation satisfies all three. Borda averages the conflict away; beam search resolves it greedily; clustering may never compare the conflicting pair at all. This is why 65% pair accuracy can coexist with 0.20 Kendall's tau.

## 7.9 Pros of this stage design

1. **Balanced supervision.** Reversals make the labels exactly 50/50.
2. **Reusable.** The same classifier orders 8 or 99 sentences.
3. **Inspectable.** The score matrix shows which pairs are uncertain.
4. **Decoder flexibility.** Borda, beam, clusters, and TF-IDF share one matrix.
5. **No hallucination.** Ordering permutes existing sentences; it cannot create text.
6. **Memory efficient.** Pairwise classification with LoRA stays within the budget.
7. **Clear ablation.** The TF-IDF baseline isolates the learned contribution.

## 7.10 Cons and failure modes

| Weakness | Cause | Evidence / effect |
| --- | --- | --- |
| **Low global order quality** | Pairwise errors compose badly. | Mean τ 0.2007 |
| **Quadratic cost** | Every unordered pair is scored. | 4,851 pairs for 99 sentences; 130.53 s for seven files |
| **Cycle inconsistency** | Local decisions are not constrained to a permutation. | Borda only approximates a solution |
| **Cluster assumption** | Topical similarity is not discourse adjacency. | Clusters τ 0.1220 |
| **Beam myopia** | Greedy adjacency can lock in an early error. | Beam τ 0.0921 |
| **Truncation** | Pairs longer than 192 subwords are cut. | Long sentences lose their endings |
| **No explicit discourse model** | mBERT sees two sentences only. | Entity/date continuity across many sentences is invisible |
| **Domain shift** | Wikipedia prose, not dialogue or lists. | Ordering may fail on the assignment samples |
| **Short-article variance** | An 8-sentence article has few constraints. | Min τ negative on some articles |

## 7.11 Alternative choices considered

| Alternative | Why rejected |
| --- | --- |
| Train directly on permutations | Needs many permutations per article; hard to score; not enough data. |
| Use sentence embeddings + sorting | Weaker than pairwise supervision; no learned before/after relation. |
| Always use beam search | Measured worse than Borda. |
| Always use clusters | Measured worse and adds assumptions. |
| Use dates/numbers/entities as rules | Useful signal, but brittle and outside the learned pipeline. |
| LLM reranking | Forbidden by the assignment. |

A stronger future design would add a transitive consistency loss or a listwise reranker on top of the pairwise scores. That is a documented improvement, not part of the shipped system.

## 7.12 Implementation map

| Function | File | Role |
| --- | --- | --- |
| `make_pairs_from_sentences` | `src/ordering/pairs.py` | consecutive pairs plus reversals |
| `SentencePairDataset` | `src/ordering/pairs.py` | tokenization of sentence pairs |
| `compute_metrics` | `src/ordering/train.py` | accuracy and macro P/R/F1 |
| `OrderScorer.score_matrix` | `src/ordering/decode.py` | symmetric pairwise probability matrix |
| `_earliness` | `src/ordering/decode.py` | mean before-score for Borda |
| `order_from_matrix` | `src/ordering/decode.py` | Borda, beam, and cluster dispatch |
| `cluster_sentences` | `src/ordering/decode.py` | TF-IDF + KMeans + medoids |
| `_beam_order` | `src/ordering/decode.py` | beam search over chains |
| `order_sentences_tfidf` | `src/ordering/decode.py` | lexical ablation |
| `kendall_tau` | `src/ordering/decode.py` | order evaluation |
| `evaluate_orderings` | `src/ordering/decode.py` | shuffled-article evaluation, one matrix reused |

<div class="pagebreak"></div>
