# 20. Reference: notation, glossary, and sources

## 20.1 Notation

| Symbol | Meaning |
| --- | --- |
| $w_1,\ldots,w_n$ | words in a document after `NULL` removal |
| $y_i$ | punctuation label for word $i$ |
| $K$ | number of punctuation classes; $K=5$ |
| $N$ | total labelled positions |
| $N_c$ | count of class $c$ |
| $w_c$ | class weight for class $c$ |
| $M_{ij}$ | probability that sentence $i$ precedes sentence $j$ |
| $n$ | number of sentences in a document |
| $\pi$ | a permutation of sentence indices |
| $s_i$ | Borda-like mean before-score for sentence $i$ |
| $\tau$ | Kendall's tau |
| $C$, $D$ | concordant and discordant pairs |
| $p_n$ | modified n-gram precision |
| $BP$ | brevity penalty |

## 20.2 Formula sheet

Punctuation accuracy:

$$
\text{acc} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat{y}_i=y_i]
$$

Boundary precision, recall, F1:

$$
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad F_1=\frac{2PR}{P+R}
$$

Capped class weight:

$$
w_c=\min\left(20,\frac{N}{K N_c}\right)
$$

Triangular window weight:

$$
a_i=\max\left(0.05,\ 1-\frac{|i-(\ell-1)/2|}{(\ell-1)/2+1}\right)
$$

Weighted probability aggregation:

$$
\bar{p}_i=\frac{\sum_w a_{i,w}p_{i,w}}{\sum_w a_{i,w}}
$$

Borda-like sentence score:

$$
s_i=\frac{1}{n-1}\sum_{j\ne i}M_{ij}
$$

Pairwise complement:

$$
M_{ji}=1-M_{ij}
$$

Kendall's tau:

$$
\tau=\frac{C-D}{\binom{n}{2}}
$$

BLEU:

$$
\text{BLEU}=BP\cdot\exp\left(\sum_{n=1}^{4}w_n\log p_n\right)
$$

## 20.3 Glossary

| Term | Definition |
| --- | --- |
| **Adapter** | A small set of trained weights added to a frozen backbone; here LoRA matrices plus a classifier. |
| **Alignment contract** | The guarantee that output item $i$ corresponds to input item $i$; used for sentences and translations. |
| **Article-level split** | Splitting by Wikipedia article so no article appears in two splits. |
| **Beam search** | Keeping the top $b$ partial sequences at each step instead of one greedy path. |
| **BLEU** | Corpus n-gram precision with a brevity penalty. |
| **Borda-like score** | Mean pairwise "before" probability used to rank sentences. |
| **Boundary F1** | F1 over the union of `sentence_end`, `question`, and `exclamation`. |
| **chrF** | Character n-gram F-score between hypothesis and reference. |
| **Class weight** | A multiplier in the loss that makes rare labels contribute more. |
| **Danda** | The Devanagari sentence-ending mark `।`. |
| **Eager attention** | A non-fused attention implementation used on MPS because SDPA with dropout is broken there. |
| **First-subword label** | Placing a word's label on its first WordPiece token; continuation tokens get `-100`. |
| **`ignore_index`** | The value `-100` that cross entropy skips. |
| **Kendall's tau** | Rank correlation between recovered and true sentence order. |
| **KMeans** | Clustering used by the TF-IDF cluster decoder. |
| **LoRA** | Low-rank adaptation; trains small matrices instead of full weights. |
| **Medoid** | The cluster member closest to the cluster centroid. |
| **MPS** | Apple Metal Performance Shaders backend. |
| **`NULL`** | Placeholder token in the assignment input, removed before processing. |
| **Pairwise classifier** | A model that scores whether A precedes B. |
| **Score matrix** | The $n\times n$ matrix of pairwise before-probabilities. |
| **SDPA** | Scaled dot-product attention; the default attention path in transformers 4.x. |
| **Subset adapter** | An adapter trained on capped examples (4,000 here), not the full recipe. |
| **TF-IDF** | Term-frequency inverse-document-frequency vectorisation; used as clustering features and as an ablation. |
| **Triangular weighting** | Down-weighting predictions near a window edge. |
| **WordPiece** | The subword tokenizer used by mBERT. |

## 20.4 Source references

### Models and methods

1. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** NAACL 2019. mBERT is the multilingual variant used here.
2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. **LoRA: Low-Rank Adaptation of Large Language Models.** ICLR 2022.
3. Mangrulkar, S., et al. **PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods.** Hugging Face library.
4. Gala, J., Chitale, P., AK, R., Gumma, V., Doddapaneni, S., Kumar, A., Nawale, J., Sujatha, A., Puduppully, R., Raghavan, V., Kumar, P., Khapra, M. M., and Dabre, R. **IndicTrans2: Towards High-Quality and Accessible Machine Translation for All 22 Scheduled Indian Languages.** 2023.
5. AI4Bharat. **IndicTransToolkit.** Preprocessing and postprocessing utilities for IndicTrans2.
6. Kudo, T. **Subword Regularization / SentencePiece**; WordPiece is the relevant mBERT tokenizer method.

### Evaluation

7. Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. **BLEU: a Method for Automatic Evaluation of Machine Translation.** ACL 2002.
8. Popović, N. **chrF: character n-gram F-score for automatic MT evaluation.** WMT 2015.
9. Kendall, M. G. **A New Measure of Rank Correlation.** Biometrika, 1938.
10. Post, M. **A Call for Clarity in Reporting BLEU Scores.** WMT 2018. `sacrebleu` follows this reproducibility convention.
11. **sacrebleu** library, version 2.6.0.

### Data and tooling

12. Wikimedia Foundation. **Wikipedia Hindi snapshot**, config `20231101.hi`, via Hugging Face `wikimedia/wikipedia`.
13. Hugging Face. **Transformers**, **Datasets**, **Tokenizers**, **Accelerate**, **Hugging Face Hub**.
14. scikit-learn: **KMeans**, **TfidfVectorizer**, **cosine_similarity**.
15. SciPy: **Kendall's tau** implementation.
16. PyTorch: **MPS backend**, **CUDA memory fraction**, **autocast / fp16**.
17. Google Chrome headless: PDF rendering for the documentation builds.
18. Python-Markdown and Pygments: HTML generation and code highlighting.

### Data-set evaluation

19. CFILT, IIT Bombay. **IITB English–Hindi parallel corpus**, via `cfilt/iitb-english-hindi` on Hugging Face. Used for translation benchmarking only.
20. Wikipedia Hindi corpus used for punctuation and ordering supervision.

## 20.5 Model identifiers

```text
bert-base-multilingual-cased
ai4bharat/indictrans2-indic-en-dist-200M
wikimedia/wikipedia  (config: 20231101.hi)
cfilt/iitb-english-hindi
```

## 20.6 Document provenance

| Item | Value |
| --- | --- |
| Repository | `nlp-zero-to-hero` |
| Project | Hindi punctuation → ordering → translation |
| Approach source | `docs/approach/*.md` |
| Builder | `scripts/build_holy_approach.py` |
| Diagram generator | `scripts/make_approach_diagrams.py` |
| Diagram assets | `docs/approach_assets/*.svg` |
| Outputs | `docs/holy-approach.html`, `docs/holy-approach.pdf` |
| Local evidence | `reports/*.json`, `models/*/training_summary.json`, `outputs/*` |
| Verification | `scripts/verify_colab_run.py` |
| Seed | 42 |
| Memory target | 1.5GB CUDA design cap |

## 20.7 Final checklist

- [ ] Every score in this document can be traced to a report.
- [ ] Subset scores are labelled as subset scores.
- [ ] Full-recipe settings are labelled as documented, not measured.
- [ ] CUDA memory is not claimed as locally verified.
- [ ] Translation metrics are tied to IITB, not assignment outputs.
- [ ] Ordering is described as weak, with tau and failure modes.
- [ ] Punctuation rare-class failures are reported.
- [ ] No LLM is used anywhere in the pipeline.
- [ ] The pipeline preserves sentence alignment.
- [ ] PDFs remain local and are not committed.
- [ ] The diagrams match the text they illustrate.
- [ ] The verifier is run before any final claim.
