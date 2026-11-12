# 11. Evaluation: measure each stage, then measure the composition

<div class="decision">

**APPROACH CHOSEN.** Evaluate stage 1 with word accuracy, boundary F1, and per-class F1; evaluate stage 2 with pair accuracy plus Kendall's tau after decoding; evaluate stage 3 with corpus BLEU and chrF on a standard benchmark. Use the same inference code as the pipeline, record dataset sizes, and report failures instead of hiding them.

</div>

The evaluation is not a single leaderboard number. The pipeline has three different tasks, and each needs a metric that matches what it is supposed to do.

## 11.1 Why one metric is not enough

| If we only report | We would miss |
| --- | --- |
| Punctuation accuracy | The model can score 0.91 by predicting `none` everywhere. |
| Pair accuracy | 65% pair accuracy can still produce a poor global order. |
| BLEU | Fluent but semantically wrong translations can score well. |
| One end-to-end score | We would not know which stage failed. |

The evaluation therefore reports a small scoreboard, not a single number.

<figure class="diagram">
<img src="approach_assets/10_scores.svg" alt="Measured local scoreboard">
<figcaption><strong>Figure 11.1.</strong> The measured scoreboard separates strong sentence-ending detection, weak comma detection, modest ordering, and usable translation. These are subset-adapter numbers.</figcaption>
</figure>

## 11.2 Punctuation metrics

### 11.2.1 Token accuracy

$$
\text{accuracy} = \frac{\text{correct word labels}}{\text{all word labels}}
$$

This is easy to interpret but dominated by `none`. It is reported, not used for model selection.

### 11.2.2 Boundary precision, recall, and F1

A prediction is a boundary if the label is `sentence_end`, `question`, or `exclamation`.

$$
P = \frac{TP}{TP+FP},\qquad
R = \frac{TP}{TP+FN},\qquad
F_1 = \frac{2PR}{P+R}
$$

This answers the practical question: did the model find the ends of sentences? It treats all three boundary types as one positive class, which is appropriate because the output format needs boundaries more than it needs the exact rare mark.

### 11.2.3 Per-class F1

For each label $c$:

$$
P_c = \frac{TP_c}{TP_c+FP_c},\qquad
R_c = \frac{TP_c}{TP_c+FN_c},\qquad
F1_c = \frac{2P_cR_c}{P_c+R_c}
$$

This exposes the imbalance. A high boundary F1 can hide a zero for `question`.

## 11.3 Ordering metrics

### 11.3.1 Pair accuracy and macro F1

The pair test set is balanced, so accuracy is meaningful:

$$
\text{accuracy} = \frac{\text{correct pair decisions}}{3{,}000}
$$

Macro F1 averages the two class F1 scores, so a model that favours one direction is penalised.

### 11.3.2 Kendall's tau

For a recovered order and the true order:

$$
\tau = \frac{C - D}{\binom{n}{2}}
$$

where $C$ is the number of concordant pairs, $D$ the number of discordant pairs, and $n$ the number of sentences. Values range from −1 (reversed) through 0 (random) to 1 (perfect).

Why tau?

- It measures the whole permutation, not isolated pairs.
- It handles articles of different lengths.
- It is interpretable: 0.2 means weak positive agreement.

The evaluator shuffles each held-out article with a seeded RNG, recovers an order, and compares against the identity order. The same score matrix is reused by every decoder, so the comparison is fair.

## 11.4 Translation metrics

### 11.4.1 BLEU

Corpus BLEU combines clipped n-gram precision with a brevity penalty:

$$
\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)
$$

where $p_n$ is the modified n-gram precision and $BP$ penalises outputs shorter than the reference. It rewards exact phrasing.

### 11.4.2 chrF

chrF computes character n-gram F-score between hypothesis and reference. It is more tolerant of morphology and word-order differences than BLEU, which matters for Hindi→English.

Both are computed with `sacrebleu` at corpus level.

## 11.5 Measured results

### 11.5.1 Punctuation

`reports/evaluation.json`, 1,500 test windows, 108,898 words:

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

The training summary's test split reports accuracy 0.8688 and boundary F1 0.8633 on the full test JSONL. The difference comes from the evaluation subset (first 1,500 windows) and the deployed sliding-window inference path. Quote the evaluation report for pipeline behaviour and the training summary for the Trainer's held-out evaluation.

### 11.5.2 Ordering

| Metric | Value |
| --- | ---: |
| Test pairs | 3,000 |
| Pair accuracy | 0.6543 |
| Macro precision | 0.6560 |
| Macro recall | 0.6543 |
| Macro F1 | 0.6534 |

Decoder comparison on 40 held-out articles:

| Decoder | Mean τ | Median τ | Min τ | Max τ |
| --- | ---: | ---: | ---: | ---: |
| **Borda** | **0.2007** | 0.2304 | −0.1775 | 0.6444 |
| Clusters | 0.1220 | 0.1081 | −0.2652 | 0.7143 |
| Beam | 0.0921 | 0.0693 | −0.2467 | 0.7143 |
| TF-IDF | 0.0631 | 0.0589 | −0.4847 | 0.5714 |

### 11.5.3 Translation

| Metric | Value |
| --- | ---: |
| Dataset | `cfilt/iitb-english-hindi` |
| Examples | 100 |
| BLEU | 34.125 |
| chrF | 59.686 |

## 11.6 Error analysis

<figure class="diagram">
<img src="approach_assets/09_errors.svg" alt="Error propagation across stages">
<figcaption><strong>Figure 11.2.</strong> Errors do not stay local. An over-segmented sentence produces bad ordering pairs, and a bad order produces translation that is fluent but out of sequence.</figcaption>
</figure>

### 11.6.1 Stage 1 errors

| Error | Evidence | Root cause |
| --- | --- | --- |
| Over-segmentation of dialogue | sample-6: 507 words → 80 sentences; sample-7: 846 words → 99 sentences | Wikipedia training sentences have at least five words; short utterances are out of distribution. |
| Low comma F1 | 0.3161 | Comma placement is style-dependent and often optional; many commas have no unique answer. |
| Zero question/exclamation F1 | 0.0 / 0.0 | Only 236 question and 153 exclamation labels in training. |
| Boundary precision below recall | 0.8539 vs 0.9167 | The model prefers to split; over-segmentation is the visible symptom. |

Concrete over-segmentation examples from `outputs/sample-6.sentences_hi.txt`:

```text
हूं।
यू।
```

These are sentence fragments, not sentences. The model is applying a boundary rule learned from longer Wikipedia sentences.

### 11.6.2 Stage 2 errors

| Error | Evidence | Root cause |
| --- | --- | --- |
| Weak global order | Mean τ 0.2007 | Pairwise errors do not compose into a consistent permutation. |
| Negative tau on some articles | Min τ −0.1775 for Borda; −0.4847 for TF-IDF | Short articles and domain shift; some recovered orders are worse than random. |
| Cluster assumption fails | Mean τ 0.1220 | Topically similar sentences are not necessarily adjacent. |
| Beam locks in early mistakes | Mean τ 0.0921 | Greedy adjacency cannot undo a bad first transition. |
| Pair model is only modestly above chance | 0.6543 accuracy | The task is hard; mBERT pair classification without discourse structure is limited. |

The gap between pair accuracy and tau is the central ordering insight:

- 65% of isolated pair decisions are correct,
- but only about 20% of the full ordering signal is recovered.

### 11.6.3 Stage 3 errors

| Error | Evidence | Root cause |
| --- | --- | --- |
| Meaning drift | Example 2 changes "As America's road planners struggle..." to "While U.S. road planners are struggling..." | Small model; beam search picks a fluent paraphrase. |
| Number changes | "The devices" → "The device" | Local sentence context is insufficient. |
| Added framing | "The usually dull arena" → "Even boring work such as..." | Model rewrites rather than translates literally. |
| Names and terms drift | Observed in samples | No glossary or constrained decoding. |
| Benchmark ≠ pipeline | BLEU is on IITB, not assignment outputs | No reference translations for assignment files. |

### 11.6.4 Composition errors

The pipeline can be wrong in a way no stage metric captures:

1. Stage 1 splits a real sentence into two fragments.
2. Stage 2 orders the fragments incorrectly.
3. Stage 3 translates each fragment fluently.
4. The final English looks grammatical but says the wrong thing.

This is why the approach documents limitations instead of presenting one aggregate score.

## 11.7 Why the weak scores are still useful

| Result | What it proves |
| --- | --- |
| Boundary F1 0.8842 | The model reliably finds most sentence endings. |
| `sentence_end` F1 0.8838 | Danda restoration works well. |
| Ordering τ 0.2007 | The learned scorer beats the TF-IDF baseline (0.0631). |
| BLEU 34.125 | The translation stage is usable on a standard benchmark. |
| Pair accuracy 0.6543 | The ordering model learned a real signal, not a label shortcut. |

The project is not claiming a solved pipeline. It is claiming a working, inspectable pipeline with measured strengths and weaknesses.

## 11.8 Statistical caution

| Limitation | Effect |
| --- | --- |
| One seed | No confidence intervals across training runs. |
| 40 ordering articles | Tau estimates are noisy; min/max are wide. |
| 100 translation examples | BLEU/chrF have sampling variance. |
| Subset adapters | Scores are not the full 25k/40k recipe. |
| Single Wikipedia snapshot | Domain and label distribution may drift. |
| Two evaluation paths for punctuation | Trainer test F1 and deployed F1 differ slightly. |

Report means and counts, not just best examples. The verifier checks that the report is present and finite, not that the scores are high.

<div class="pagebreak"></div>

# 12. Verification: the checks that keep the claims honest

<div class="decision">

**APPROACH CHOSEN.** Use a fail-fast verifier that checks data counts, adapter files, report schemas, metric thresholds, output alignment, and notebook structure. Run it on the full pipeline before making claims. Treat the verifier as part of the approach, not an afterthought.

</div>

## 12.1 Why a verifier

A notebook can appear to run while silently skipping a stage. A report can contain stale numbers. An output file can have a different number of Hindi and English lines. The verifier makes those failures loud.

`scripts/verify_colab_run.py` is the executable contract. It checks that the repository actually contains what the documentation says it contains.

## 12.2 What the verifier checks

| Area | Checks |
| --- | --- |
| Data summary | split sizes 2,700/150/150; 25,000 punctuation train windows; 40,000 ordering train pairs; 40 eval articles |
| Punctuation data | JSONL row counts; words/labels alignment; label vocabulary; window length bounds |
| Ordering data | pair counts; binary labels; approximately balanced classes |
| Adapters | adapter directories, config files, tokenizer files, training summaries |
| Reports | required keys, status `ok`, finite metrics, sample counts |
| Thresholds | punctuation and ordering quality floors |
| Outputs | all seven stems; expected file names; Hindi/English line alignment; groups of five |
| Notebooks | expected cell counts and structure |

## 12.3 Thresholds

| Metric | Minimum |
| --- | ---: |
| Punctuation accuracy | 0.85 |
| Punctuation boundary F1 | 0.85 |
| `none` F1 | 0.85 |
| `sentence_end` F1 | 0.80 |
| Ordering pair accuracy | 0.58 |
| Ordering macro F1 | 0.55 |
| Borda Kendall tau | 0.08 |
| Train boundary F1 | 0.75 |
| Train ordering accuracy | 0.58 |
| BLEU | 30 |
| chrF | 50 |

These are floors, not targets. They catch a broken run, not a mediocre one.

## 12.4 How to run it

```bash
python scripts/verify_colab_run.py
```

The full local run passed 150/150 checks. The `--skip-model-smoke` path passed 142 checks; the remaining eight are smoke-test checks that require a separate quick-run artifact.

## 12.5 What the verifier does not prove

- It does not prove the model is good; it proves the run is structurally consistent.
- It does not re-train the models.
- It does not validate translation quality beyond the recorded metrics.
- It does not check that the 1.5GB CUDA cap was exercised.
- It does not replace human inspection of samples.

That boundary is deliberate. A verifier that claims more than it checks is worse than no verifier.

## 12.6 Review questions the evaluation answers

| Question | Evidence |
| --- | --- |
| Did you use an LLM? | `no_llm: true`; only mBERT and IndicTrans2. |
| How good is punctuation? | Boundary F1 0.8842; sentence_end F1 0.8838; comma F1 0.3161. |
| Why is comma weak? | Style ambiguity and class imbalance. |
| How good is ordering? | Pair accuracy 0.6543; Borda tau 0.2007. |
| Why not use clusters? | Measured tau 0.1220, worse than Borda. |
| How good is translation? | BLEU 34.125, chrF 59.686 on 100 IITB pairs. |
| Is the pipeline aligned? | Sentence-level contract plus output alignment checks. |
| Is it reproducible? | Seed 42, pinned dataset, verifier, recorded reports. |
| What is the biggest weakness? | Global ordering; documented with tau and failure modes. |
| Are the scores from the full recipe? | No; subset adapters. This is stated everywhere it matters. |

<div class="pagebreak"></div>
