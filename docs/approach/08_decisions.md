# 13. Decision log: every major choice and the alternative it beat

<div class="decision">

**APPROACH CHOSEN.** Record decisions as explicit trade-offs, not as a list of technologies. Each choice below is tied to a constraint, a rejected alternative, and a measured or structural consequence. The decision spine is the design; the code is its implementation.

</div>

<figure class="diagram">
<img src="approach_assets/12_decisions.svg" alt="Decision spine">
<figcaption><strong>Figure 13.1.</strong> The design choices form a chain: preserve Hindi text, keep the model small, decompose the task, isolate memory, and evaluate each stage separately.</figcaption>
</figure>

## 13.1 Decision summary

| # | Decision | Chosen | Main alternative rejected | Why |
| ---: | --- | --- | --- | --- |
| 1 | Task decomposition | Three stages | End-to-end generation | No aligned triplets; isolation; small models |
| 2 | Punctuation formulation | Word-level 5-class classification | Seq2seq punctuation | Guarantees word preservation |
| 3 | Backbone | mBERT | MuRIL / IndicBERT / larger encoder | Fits budget; strong multilingual baseline; known tooling |
| 4 | Adaptation | LoRA | Full fine-tuning | Tiny checkpoints; low memory |
| 5 | Punctuation decoding | Overlapping windows + triangular weights | One window / uniform average | More context; robust edge predictions |
| 6 | Class imbalance | Capped inverse-frequency weights | Resampling / ignore rare classes | Keeps all data; simple; auditable |
| 7 | Ordering formulation | Pairwise classifier | Listwise / generative permutation | Balanced data; reusable for any length |
| 8 | Ordering decoder | Borda-like mean score | Beam / clusters / TF-IDF | Measured best tau |
| 9 | Translation model | IndicTrans2 200M | NLLB / mBART / LLM | Indic-specific; small; strong |
| 10 | Translation unit | Sentence | Whole document / paragraph | Exact alignment; required output format |
| 11 | Data source | Hindi Wikipedia | Synthetic / LLM-generated | Natural supervision; assignment forbids LLM |
| 12 | Data split | Article-level | Window-level / sentence-level | Prevents leakage |
| 13 | Memory strategy | Load one stage at a time | Load all models | Peak equals largest model |
| 14 | Failure strategy | Loud failure, no fallback | Silent fallback model | Reproducibility and auditability |
| 15 | Evaluation | Per-stage metrics + composition checks | One aggregate score | Localises failures |

## 13.2 Decision 1: decompose into three stages

### Context

The assignment asks for punctuation, ordering, translation, and grouping from raw Hindi tokens. End-to-end training would require aligned `(raw tokens, ordered English sentences)` triples, which are not available.

### Chosen

Three learned/rule stages with explicit text contracts:

```text
tokens -> punctuated sentences -> ordered sentences -> English sentences
```

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| One seq2seq model from tokens to English | No aligned training data; cannot inspect intermediate failures; one model would have to learn three tasks at once. |
| Two stages: punctuate+translate, then order English | Ordering English changes the learned task; translation errors then affect ordering. |
| Translation first, then punctuation/ordering | Punctuation is language-specific; translating unpunctuated text is harder; order would be recovered in English, not Hindi. |

### Consequence

- Each stage can be evaluated and replaced independently.
- Errors compound, but they are localisable.
- The pipeline is longer than a single model but far more auditable.

## 13.3 Decision 2: punctuation as token classification

### Context

The input already contains every word. The task is to insert marks, not to generate text.

### Chosen

Five-class classification per word, with labels attached to the first subword.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| Seq2seq punctuation | Could rewrite words; slower; harder alignment; larger model. |
| Rule-based punctuation | Brittle; cannot use contextual cues; poor on ambiguous commas. |
| LLM prompting | Forbidden; non-deterministic; opaque. |
| Span prediction | More complex for the same output; no clear advantage. |

### Consequence

- Word preservation is guaranteed by construction.
- Evaluation is exact at the word level.
- Rare classes remain hard because the model only chooses among five labels; it cannot invent a better training distribution.

## 13.4 Decision 3: mBERT as the shared backbone

### Context

The project needs a Hindi-capable encoder that fits a 1.5GB budget and runs on Colab and Apple Silicon.

### Chosen

`bert-base-multilingual-cased`.

### Rejected alternatives

| Alternative | Pros | Why rejected |
| --- | --- | --- |
| MuRIL | Indic-specific; strong Hindi | Larger; different tokenizer; tooling less predictable for this project. |
| IndicBERT | Indic-specific | Masked-LM checkpoint; classification task support is less standard. |
| XLM-R base | Strong multilingual | Larger than mBERT base; more memory. |
| DistilmBERT | Smaller and faster | Lower capacity; less proven for Hindi punctuation. |
| A larger encoder | Better ceiling | Does not fit the budget; contradicts the constraint. |

### Consequence

- One backbone family for both learned stages.
- 178M parameters, but only about 0.33% trained through LoRA.
- The model is not the best possible Hindi encoder; it is the best fit for the constraints.

## 13.5 Decision 4: LoRA instead of full fine-tuning

### Context

The budget is 1.5GB and the deliverable includes adapters, not a training cluster.

### Chosen

LoRA rank 16, alpha 32, dropout 0.1, on query and value projections.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| Full fine-tuning | Much more memory; larger checkpoints; slower; unnecessary for this scale. |
| Frozen encoder + linear head | Too weak for punctuation and pair classification. |
| Adapters (Houlsby) | More parameters and complexity than LoRA for the same goal. |
| Prompt tuning | Less mature for BERT classification; harder to reproduce. |

### Consequence

- Punctuation: 593,669 trainable parameters (0.3338%).
- Ordering: 591,362 trainable parameters (0.3314%).
- Adapter directories are small enough to ship.
- The ceiling is slightly lower than full fine-tuning; this is an accepted trade-off.

## 13.6 Decision 5: overlapping punctuation windows

### Context

mBERT has a finite input length; documents are longer. A single window loses context; fixed non-overlapping windows create edge artifacts.

### Chosen

64-word windows, 40-word stride, triangular weighting, minimum weight 0.05.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| One window per document | Truncation loses most of the document. |
| Non-overlapping windows | Predictions at boundaries have little context. |
| Uniform averaging | Treats unreliable edge predictions as equal to central ones. |
| Very large windows | More memory; padding waste; no guarantee of better accuracy. |

### Consequence

- A word can be seen in several windows.
- Edge predictions are down-weighted, not discarded.
- Inference cost grows with overlap but remains small: 2.31 seconds for all seven files.

## 13.7 Decision 6: capped class weights

### Context

`none` is 90.947% of labels. Rare classes are almost absent.

### Chosen

$$
w_c = \min\left(20,\frac{N}{K N_c}\right)
$$

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| No weighting | Model collapses to `none`; boundaries are missed. |
| Random oversampling | Repeats the same rare examples; risks overfitting; changes epoch semantics. |
| Undersampling `none` | Throws away most of the data. |
| Focal loss | Plausible, but adds another hyperparameter and was not measured. |
| Ignore rare classes | Hides the problem instead of reporting it. |

### Consequence

- Rare classes contribute more gradient.
- The cap prevents question/exclamation from destabilising training.
- Weights alone cannot create diversity; rare-class F1 remains zero.

## 13.8 Decision 7: pairwise ordering

### Context

Ordering is a permutation problem. Direct permutation training needs many labelled permutations.

### Chosen

Binary before/after classification on consecutive sentence pairs, with reversals.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| Listwise classifier | Needs a fixed maximum length; harder to train; less reusable. |
| Seq2seq permutation | Can drop or invent sentences; needs permutation data. |
| Sentence embeddings + sort | Weaker; no learned before/after relation. |
| LLM ordering | Forbidden; expensive; non-deterministic. |

### Consequence

- Labels are exactly balanced.
- The classifier works for any document length.
- Global consistency becomes a decoding problem.

## 13.9 Decision 8: Borda-like decoding

### Context

Pairwise scores are local and can be inconsistent. The decoder must produce a permutation.

### Chosen

Sort by mean before-score:

$$
s_i = \frac{1}{n-1}\sum_{j \ne i} M_{ij}
$$

### Rejected alternatives

| Alternative | Measured tau | Why rejected |
| --- | ---: | --- |
| Clusters | 0.1220 | Hard topical assumption; worse than Borda. |
| Beam | 0.0921 | Greedy adjacency locks in early errors. |
| TF-IDF | 0.0631 | Lexical similarity is not discourse order. |
| Integer linear programming | Not measured | Potentially better global consistency, but more complex and slower. |

### Consequence

- Borda is the measured winner and the default.
- It is cheap: sorting after the matrix is built.
- It still produces only weak global order; the decoder choice cannot fix a weak scorer.

## 13.10 Decision 9: IndicTrans2 200M

### Context

The translation stage needs Hindi→English quality within the budget, with no LLM.

### Chosen

`ai4bharat/indictrans2-indic-en-dist-200M`.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| NLLB-200 | Larger; less Indic-specialised at the size that fits. |
| mBART / mT5 | Weaker zero-shot Hindi→English; would need fine-tuning. |
| OPUS-MT | Smaller but lower quality. |
| Larger IndicTrans2 | Better quality but does not fit the memory target. |
| LLM | Forbidden. |

### Consequence

- Strong Indic-specific translation at a small size.
- Extra dependency on IndicTransToolkit.
- Gated upstream model requires a local mirror on this machine.

## 13.11 Decision 10: sentence-level translation

### Context

The output format is sentence-level Hindi and English, plus groups of five.

### Chosen

Translate one sentence at a time, batched.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| Whole document | Long inputs; alignment harder; memory spikes. |
| Paragraph chunks | Better context but needs chunk alignment; can split discourse. |
| Translate before ordering | Changes the ordering task; errors compound differently. |

### Consequence

- Exact one-to-one alignment.
- Cross-sentence context is lost; pronouns and discourse links can drift.
- Grouping stays a pure positional operation.

## 13.12 Decision 11: Wikipedia as the data source

### Context

The assignment provides raw inputs but no labels. The project must create supervision without an LLM.

### Chosen

`wikimedia/wikipedia`, config `20231101.hi`.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| LLM-generated labels | Forbidden; circular; hides the pipeline. |
| Manually labelled data | Too small; not reproducible here. |
| Other web corpora | Licensing and preprocessing variance; Wikipedia is cleaner and pinned. |
| Assignment files as training data | No labels; would leak test inputs. |

### Consequence

- Natural punctuation and order supervision.
- Domain mismatch with dialogue and informal text.
- Snapshot drift is controlled by pinning the config.

## 13.13 Decision 12: article-level splits

### Context

Windows and sentence pairs come from articles. Splitting at the window level would leak document context.

### Chosen

Shuffle article ids with seed 42; 90/5/5 split.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| Window-level split | Neighbouring overlapping windows would appear in train and test. |
| Sentence-level split | Same article in multiple splits; leakage. |
| Random pair split | Reversals of the same pair could land in different splits. |

### Consequence

- No article appears in two splits.
- The test set is smaller (150 articles) but honest.
- Ordering evaluation uses complete held-out articles.

## 13.14 Decision 13: sequential model loading

### Context

The 1.5GB budget is smaller than the sum of the three models.

### Chosen

Load one stage, process all files, free it, load the next.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| Load all models at once | Exceeds the budget. |
| Reload the model per file | Wasteful; slower; more memory churn. |
| Quantise everything | Quality risk; not needed once stages are sequential. |

### Consequence

- Peak memory is the largest single stage.
- Runtime includes model load/unload overhead, which is small relative to ordering.
- The pipeline is simple to reason about.

## 13.15 Decision 14: no fallback

### Context

A fallback can make a demo run when a model is missing.

### Chosen

Fail loudly with a training or setup instruction.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| Rule-based fallback | Hides model failure; produces different behaviour. |
| Second translation model | Doubles memory; complicates claims. |
| Silent skip of a stage | Output shape changes; results become incomparable. |

### Consequence

- Failures are visible.
- Reproduction is deterministic.
- The pipeline cannot pretend to work when a component is missing.

## 13.16 Decision 15: per-stage evaluation

### Context

One aggregate score cannot tell us whether punctuation, ordering, or translation failed.

### Chosen

Separate metrics plus composition checks.

### Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| One end-to-end BLEU | Requires reference translations for assignment files; hides stage failures. |
| Human evaluation only | Not reproducible; expensive; small sample. |
| Model loss only | Not interpretable; does not measure deployment behaviour. |

### Consequence

- The scoreboard localises weaknesses.
- The report records dataset sizes and paths.
- The verifier checks structure and thresholds separately.

<div class="pagebreak"></div>

# 14. Pros, cons, and the honest claim

## 14.1 What the system does well

| Strength | Why it matters |
| --- | --- |
| Preserves every Hindi word | Stage 1 only assigns labels; stages 2–3 permute and translate. |
| Strong sentence-ending restoration | Boundary F1 0.8842; sentence_end F1 0.8838. |
| Learned ordering beats lexical baseline | Borda tau 0.2007 vs TF-IDF 0.0631. |
| Usable translation quality | BLEU 34.125, chrF 59.686 on 100 IITB pairs. |
| Fits the memory target by construction | One model at a time; LoRA adapters. |
| Fully inspectable | No LLM; every stage has a small interface. |
| Reproducible | Seed 42, pinned data, recorded reports, verifier. |
| Honest about weaknesses | Comma and rare-class F1, ordering tau, domain mismatch all documented. |

## 14.2 What the system does not do well

| Weakness | Evidence | Consequence |
| --- | --- | --- |
| Comma restoration is weak | F1 0.3161 | Many commas are missing or misplaced. |
| Question/exclamation are effectively unsupported | F1 0.0 | The model rarely predicts them. |
| Global ordering is weak | Mean tau 0.2007 | Documents are partially reordered, not fully reconstructed. |
| Dialogue is over-segmented | 507 words → 80 sentences; 846 → 99 | Short utterances become sentence fragments. |
| Translation drifts | BLEU 34, sample paraphrases | Meaning can change while fluency stays high. |
| No cross-sentence context in translation | Sentence-level interface | Pronouns and discourse links can be wrong. |
| No CUDA memory measurement locally | MPS no-op | The 1.5GB cap is implemented but not verified on this machine. |
| Subset adapters | 4,000 examples each | Scores are not the full 25k/40k recipe. |

## 14.3 The claim the project can defend

> A constrained, inspectable Hindi→English pipeline that restores sentence boundaries well, restores commas modestly, improves sentence order over a lexical baseline but does not fully recover document order, and translates with a small Indic-specific model at usable quality — while respecting a no-LLM rule and a 1.5GB CUDA budget design.

## 14.4 The claims the project must not make

| Overclaim | Why it is false |
| --- | --- |
| "The pipeline reconstructs documents accurately." | Mean tau 0.2007 is weak. |
| "The punctuation model handles questions and exclamations." | Both F1 are 0.0. |
| "The models were trained on the full 25k/40k data." | Shipped adapters are 4k subsets. |
| "The 1.5GB CUDA cap was verified." | It was not exercised locally on CUDA. |
| "BLEU 34 measures the assignment output." | It measures 100 IITB pairs. |
| "The system uses no pretrained translation model." | It uses IndicTrans2; "no LLM" is the actual claim. |
| "The pipeline works on any Hindi text." | Domain mismatch and dialogue failures are observed. |

## 14.5 Constraint-by-constraint compliance

| Constraint | How the design complies | Caveat |
| --- | --- | --- |
| No LLM solution | Only mBERT classifiers and IndicTrans2 | IndicTrans2 is a pretrained MT model, not an LLM pipeline. |
| 1.5GB GPU budget | LoRA + sequential loading + CUDA fraction cap | Cap is not locally measured on CUDA. |
| Hindi/Devanagari input | Whitespace tokenization; danda-aware labels | Parser is tuned to Wikipedia prose. |
| Sentence and group output | Sentence contract + groups of five | Grouping is positional, not semantic. |
| Reproducibility | Seed, pinned snapshot, reports, verifier | One seed; no confidence intervals. |

<div class="pagebreak"></div>

# 15. Improvements: what would actually move the numbers

The improvements are ordered by expected impact, not by implementation ease.

## 15.1 Priority 1: fix ordering

Ordering is the weakest stage and the largest source of end-to-end error.

| Improvement | Why it should help | Cost |
| --- | --- | --- |
| Train on the full 40,000 pairs | The shipped adapter sees only 4,000. | More compute; still within the design. |
| Hard negative mining | Random reversals are easy; hard negatives teach discourse cues. | Need a mining pass. |
| Listwise reranker on top of pair scores | Directly optimises a permutation instead of isolated pairs. | New model and training data. |
| Transitive consistency loss | Penalises cycles such as $A \prec B \prec C \prec A$. | More complex training. |
| Integer linear programming decoder | Finds a globally consistent order from pair probabilities. | Slower; more engineering. |
| Entity/date/number features | Strong ordering cues that mBERT may miss. | Feature engineering; rule risk. |
| Paragraph-level grouping before ordering | Reduces pair count and adds local context. | Grouping errors can propagate. |

The cheapest high-impact step is full-data training plus hard negatives. The most principled step is a listwise or ILP decoder on top of the existing scores.

## 15.2 Priority 2: improve punctuation, especially commas and rare classes

| Improvement | Why it should help | Cost |
| --- | --- | --- |
| Full 25,000-window training | The shipped punctuation adapter sees 4,000. | More compute. |
| Add dialogue/news data | The current corpus is encyclopedic. | Licensing and preprocessing. |
| Targeted comma data | Comma F1 is 0.3161. | Annotation or careful mining. |
| Rare-class oversampling with augmentation | Only 236 questions and 153 exclamations. | Risk of overfitting; must be measured. |
| Focal loss or label smoothing | May improve rare-class learning. | Another hyperparameter. |
| CRF or span-level decoding | Enforces label transitions. | More complex inference. |
| Post-processing rules for question marks | Questions often have interrogative words. | Rule brittleness. |

Class weights alone have already been tried; the bottleneck is rare-class diversity, not loss weighting.

## 15.3 Priority 3: improve translation quality and consistency

| Improvement | Why it should help | Cost |
| --- | --- | --- |
| Paragraph-level context | Resolves pronouns and discourse links. | Alignment complexity. |
| Glossary / constrained decoding | Keeps names and technical terms consistent. | Domain work. |
| Larger IndicTrans2 checkpoint | Better quality. | Breaks the small-memory target. |
| Two-pass consistency check | Fixes entity and number drift. | Extra inference. |
| Human evaluation on assignment samples | Measures what BLEU cannot. | Manual effort. |

## 15.4 Priority 4: strengthen evaluation

| Improvement | Why it should help | Cost |
| --- | --- | --- |
| Multiple seeds | Confidence intervals for tau and F1. | More training runs. |
| More ordering articles | Reduce tau variance. | More evaluation time. |
| Larger translation sample | Reduce BLEU/chrF variance. | More inference. |
| Pipeline-level reference set | Measure end-to-end meaning, not just stages. | Manual references. |
| Error annotation taxonomy | Quantify over-segmentation vs ordering vs translation errors. | Manual review. |
| Peak CUDA memory logging | Verify the 1.5GB claim. | Requires CUDA hardware. |

## 15.5 Priority 5: engineering and reproducibility

| Improvement | Why it should help | Cost |
| --- | --- | --- |
| Cache pairwise scores | Avoid recomputation across decoders. | Storage and invalidation logic. |
| Parallelise ordering batches | Reduce the 130-second ordering time. | More memory or devices. |
| Add confidence estimates | Flag uncertain boundaries and pairs. | Calibration work. |
| Package adapters with checksums | Detect corrupted checkpoints. | Small. |
| Add a one-command reproduction script | Reduce setup errors. | Small. |
| Record exact package versions in reports | Stronger reproducibility. | Small. |

## 15.6 What not to do

| Temptation | Why to avoid it |
| --- | --- |
| Add an LLM to "fix" ordering or translation | Violates the assignment and defeats the purpose. |
| Hide rare-class failures behind accuracy | Dishonest and easy to detect. |
| Report subset scores as full-recipe scores | Misrepresents the training run. |
| Add a fallback model to make demos pass | Hides failures and complicates claims. |
| Tune on the test set | Invalidates the evaluation. |
| Use the assignment files as training data | Leakage and no labels. |
| Overfit the approach document to the scoreboard | The document must explain the design, not just the numbers. |

## 15.7 Improvement decision rule

A change is worth shipping only if:

1. it addresses a measured weakness,
2. it respects the no-LLM and memory constraints,
3. it can be evaluated on held-out data,
4. it does not make the pipeline less inspectable,
5. its claim can be stated without qualifiers that contradict the evidence.

This rule is why the current design ships Borda, documents ordering as weak, and does not pretend that class weights solved the rare-class problem.

<div class="pagebreak"></div>
