<div class='pagebreak'></div>

# Part II — The project in depth

## 15. Repository tour

```text
nlp-zero-to-hero/
|-- data/
|   |-- raw/                     # sample-1..7.txt (space-separated words, NULL tokens)
|   `-- processed/               # generated JSONL datasets (git-ignored)
|-- src/
|   |-- constants.py             # paths, model ids, labels, hyper-parameters
|   |-- utils.py                 # seeding, devices, memory, Devanagari helpers
|   |-- data_prep.py             # Wikipedia -> punctuation + ordering datasets
|   |-- punctuation/
|   |   |-- dataset.py           # word -> subword label alignment
|   |   |-- train.py             # mBERT + LoRA token classifier
|   |   `-- predict.py           # sliding-window inference
|   |-- ordering/
|   |   |-- pairs.py             # before/after pair dataset
|   |   |-- train.py             # mBERT + LoRA pair classifier
|   |   `-- decode.py            # Borda, clusters, beam, Kendall's tau
|   |-- translation/
|   |   `-- indictrans.py        # IndicTrans2 wrapper
|   |-- pipeline.py              # end-to-end orchestration
|   `-- evaluate.py              # metrics and reports
|-- notebooks/
|   |-- colab_end_to_end.ipynb   # 28-cell full Colab recipe
|   |-- pipeline_demo.ipynb      # 13-cell executed local demo
|   `-- lesson_01..03.ipynb      # early generic NLP exercises
|-- baseline/                    # legacy rule-based version (comparison only)
|-- lessons/                     # preprocessing and TF-IDF exercises
|-- outputs/                     # generated Hindi/English sentence and group files
|-- reports/                     # JSON metrics
|-- scripts/colab_setup.sh       # dependency installer
|-- requirements-colab.txt       # verified GPU/Colab pins
|-- requirements.txt             # local CPU environment
`-- README.md
```

### 15.1 What each source file does

| File | Responsibility | Key detail |
| --- | --- | --- |
| `src/constants.py` | Single source of truth for paths, model names, labels, hyperparameters | all paths are absolute, derived from the repo root |
| `src/utils.py` | Reusable helpers | `set_seed`, `get_device`, `attention_implementation`, `limit_gpu_memory`, `sliding_windows`, `window_weight`, `strip_punctuation`, `split_into_sentences`, `group_lines` |
| `src/data_prep.py` | Stream Wikipedia, parse articles, build windows and pairs, split by article | never uses a word-character regex; shuffles sentences inside windows |
| `src/punctuation/dataset.py` | Align word labels to subwords | first subword labelled, rest `-100` |
| `src/punctuation/train.py` | Train the token classifier | LoRA, class-weighted loss, best boundary F1 |
| `src/punctuation/predict.py` | Predict one label per word | overlapping windows + triangular weights |
| `src/ordering/pairs.py` | Build balanced pairs | positive `(A,B,1)` and negative `(B,A,0)` |
| `src/ordering/train.py` | Train the pair classifier | `[CLS] A [SEP] B [SEP]`, best accuracy |
| `src/ordering/decode.py` | Matrix -> order | Borda, clusters, beam, TF-IDF ablation, tau |
| `src/translation/indictrans.py` | Hindi -> English | IndicProcessor, tags, batch generation |
| `src/pipeline.py` | Run all stages | load/free one model at a time; write four files per sample |
| `src/evaluate.py` | Measure all stages | punctuation, ordering, translation; JSON report |

### 15.2 The `baseline/` folder

The baseline is the first, rule-based attempt:

| File | What it does |
| --- | --- |
| `hindi_utils.py` | Unicode-safe `str.split` tokenizer that removes `NULL` |
| `segment.py` | Dynamic-programming sentence splitter using verb endings, starters, `NULL` hints, and a length prior |
| `order.py` | TF-IDF (char 2-4 grams) + KMeans + greedy ordering, with a synthetic Kendall's tau check |
| `translate.py` | `Helsinki-NLP/opus-mt-hi-en` MarianMT wrapper |
| `pipeline.py` | End-to-end rule baseline |

It is kept for comparison and to show the evolution of the work. The final pipeline replaces every heuristic with a trained model. The baseline is **not** used by `src/`.

### 15.3 The `lessons/` folder

These are early, generic NLP exercises, not part of the final assignment:

- `lesson_01_basics.ipynb`: environment smoke test (Python, NumPy, pandas).
- `lesson_02_tokenization.ipynb`: NLTK word and sentence tokenization on English.
- `lesson_03_preprocessing.ipynb`: cleaning, stopword removal, stemming, lemmatization.
- `lessons/preprocessing.py`: `clean_text`, `tokenize`, `stem_tokens`, `lemmatize_tokens`, `sentence_split`.
- `lessons/train.py`: TF-IDF + logistic regression on `data/raw/dataset.csv` (a generic text-classification exercise).

They document the learning path: before building transformers, the project established the classical NLP basics.

## 16. Environment and dependency saga

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Pin an exact, verified combination of libraries rather than chasing “latest”. Write the compatibility reasons into the requirements file itself. Install IndicTransToolkit separately because it may need a source build. This turns a fragile stack into a reproducible one.
</div>

The final verified combination is pinned in `requirements-colab.txt`:

```text
transformers==4.44.2
tokenizers==0.19.1
datasets==3.6.0
huggingface-hub>=0.24,<1.0
accelerate>=0.34,<2.0
peft==0.13.2
sentencepiece>=0.1.99
sacremoses
sacrebleu
protobuf
scikit-learn
scipy
numpy
pandas
tqdm
```

plus `IndicTransToolkit==1.1.1`, installed by `scripts/colab_setup.sh` because it may need a source build.

### 16.1 Why each pin exists

| Pin | Reason |
| --- | --- |
| `transformers==4.44.2` | transformers 5 removes modules that IndicTrans2's remote code and IndicTransToolkit import. 4.44.x is the verified compatible line. |
| `tokenizers==0.19.1` | version required by transformers 4.44.x; mismatches cause fast-tokenizer errors. |
| `datasets==3.6.0` | datasets 4/5 require `huggingface-hub>=1.31`, which transformers 4.44 does not support. |
| `huggingface-hub>=0.24,<1.0` | keeps the hub client compatible with transformers 4.44 and datasets 3.6. |
| `accelerate>=0.34,<2.0` | Trainer integration; newer majors can change APIs. |
| `peft==0.13.2` | newer peft needs `torch.distributed.tensor`, which is absent from some macOS torch builds. |
| `sentencepiece` | tokenizer backend for IndicTrans2. |
| `sacremoses` | text normalization used by some tokenizers. |
| `sacrebleu` | BLEU/chrF evaluation. |

> **The transformers 5 problem in one sentence.** The model code was written against transformers 4.x module paths; transformers 5 reorganized those paths, so importing the model fails even though the model weights are fine. Pinning 4.44.2 is the pragmatic fix.

### 16.2 The MPS attention problem

On Apple Silicon, PyTorch's MPS backend does not implement dropout inside `scaled_dot_product_attention` in this torch build. Transformers 4.x uses SDPA by default, so training fails or produces incorrect results. The project detects MPS and switches to **eager attention**:

```python
def attention_implementation():
    import torch
    if torch.cuda.is_available():
        return None
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return 'eager'
    return None
```

Both training scripts pass `attn_implementation='eager'` when needed. On CUDA, the faster default is kept.

### 16.3 The gated IndicTrans2 problem

The official `ai4bharat/indictrans2-indic-en-dist-200M` is gated. Without a Hugging Face token, loading returns 401. For local verification, the project uses a merged, ungated mirror at `models/translation/indictrans2-indic-en-200M/`. The wrapper accepts either a Hub id or a local path, so no code change is needed. On Colab, authenticate and accept the license, then use the official id. The code never silently falls back to a different translation model; if the model cannot load, it raises an error.

### 16.4 The data-preparation thread hang

Streaming Wikipedia with `datasets` can leave non-daemon threads alive after the script finishes, so the process may appear to hang at exit. The data is already written. In practice, run data preparation in the background and poll the log, or interrupt once the summary is written. This is a tooling quirk, not a logic bug.

### 16.5 Two local environments

The repository has two local environments:

- `.venv/` (Python 3.13): CPU libraries, Jupyter, scikit-learn. Used for `setup_check.py` and CPU-only CLIs.
- `.venv.backup-20261109-124948/` (Python 3.12): torch 2.15, transformers 4.44.2, peft 0.13.2, datasets 3.6.0. Used for the full deep-learning pipeline on MPS.

The Jupyter kernel `nlp-zero-to-hero-mps` points at the second environment. The `nlp-zero-to-hero` kernel points at the first. Use the MPS kernel for the demo notebook.

### 16.6 Setup commands

Fresh local setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements-colab.txt
python -m pip install 'IndicTransToolkit==1.1.1'
```

If the toolkit wheel fails, build from source as `scripts/colab_setup.sh` does:

```bash
git clone --depth 1 https://github.com/VarunGumma/IndicTransToolkit.git /tmp/IndicTransToolkit
python -m pip install --editable /tmp/IndicTransToolkit --no-build-isolation
```

> **Check yourself.** Why not just install the latest versions of everything? *Because these libraries are coupled: transformers constrains tokenizers, datasets constrains huggingface-hub, and peft constrains torch. The “latest” combination is frequently incompatible, especially across major versions. Pinning is how you make a pipeline reproducible.*

## 17. Data preparation deep dive

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Create supervision by hiding information that Wikipedia already contains. Punctuation is removed and stored as word labels; sentence order is shuffled inside windows; ordering pairs come from consecutive sentences. Split by article, cap examples for tractability, and never synthesize labels with an LLM.
</div>

`src/data_prep.py` turns raw Wikipedia into two datasets. It is worth understanding line by line because the model can only be as good as its labels.

### 17.1 Streaming Wikipedia

```python
stream = load_dataset(WIKI_DATASET, WIKI_CONFIG, split='train', streaming=True)
```

- `wikimedia/wikipedia` with config `20231101.hi` is the November 2023 Hindi Wikipedia dump.
- `streaming=True` means rows are downloaded on demand; the full dump is never stored.
- `scan_multiplier=3` means the loader scans up to 9,000 pages to keep 3,000 usable articles. Some pages are lists, stubs, or otherwise unusable.

### 17.2 Parsing an article

```python
ARTICLE_PATTERN = re.compile(r'(?<=[।॥?!])\s+')
```

Steps in `parse_article`:

1. Remove citation markers like `[12]` with a regex.
2. Collapse whitespace.
3. Split after sentence-final marks using the lookbehind regex.
4. For each token, call `strip_punctuation()` to separate the word from its trailing mark.
5. Keep tokens that contain Devanagari or a digit. This drops stray Latin words, URLs, and markup.
6. Accumulate words until a boundary label. If the sentence length is between 5 and 40 words, keep it. Otherwise discard it.
7. If the article yields fewer than 3 sentences, discard the article.

> **Why discard short sentences?** Wikipedia has many fragments (headings, list items, captions). Training on them would teach the model that a sentence can be two words long, which is not true for the assignment's prose. The 5-word minimum is a data-cleaning choice, and it also explains why dialogue-like samples are over-segmented at inference.

### 17.3 Punctuation windows

`make_punctuation_windows` slides over the article's sentences:

- Start at sentence `index`.
- Add sentences until the total would exceed 96 words or reaches at least 64 words.
- Keep the window if it has 32-96 words and at least 2 sentences.
- **Shuffle the sentences inside the window** with the seeded RNG.
- Concatenate their words and labels.
- Advance by at least 40 words (stride), moving whole sentences.

Why shuffle? The assignment input has shuffled sentences. If training windows kept the original order, the model could learn cross-sentence continuity cues that will not exist at inference. Shuffling forces the model to rely on local word context, which is the right signal for punctuation.

The window is measured in **words**, not subwords. A 64-word window may become 100+ subword tokens; `MAX_SEQ_LENGTH=128` truncates the tail. The first-subword label alignment ensures the remaining words still have labels.

### 17.4 Splitting by article

Article IDs are shuffled with a fixed seed, then split 90/5/5. The same split is used for punctuation and ordering. This prevents leakage and makes the two stages comparable.

### 17.5 Capping examples

Training punctuation is capped at 25,000 windows, ordering at 40,000 pairs. Evaluation is capped at 3,000 pairs (and the punctuation test is used in full unless `--max-punct-examples` is set). The cap keeps training time bounded and makes runs comparable.

> **Subtle detail.** The punctuation loop breaks out of the article loop once it has 1.2x the window limit, then shuffles and caps. This means the training set is a random sample of the first ~30,000 windows, not a random sample of all possible windows. For a course project this is fine; for a production system you would stream all articles and sample uniformly.

### 17.6 Ordering pairs

`src/ordering/pairs.py:13-49` builds pairs from consecutive sentences:

- Choose up to 30 positions per article, evenly spaced if the article is longer.
- For each position, take `(sentences[i], sentences[i+1])`.
- Emit a positive example `(A, B, label=1)`.
- Emit a negative example `(B, A, label=0)`.

This gives a balanced binary dataset and teaches the model the direction of local narrative flow. It does not teach global order directly; the decoder must combine local preferences into a global permutation.

### 17.7 The held-out ordering articles

For Kendall's tau, the project saves 40 test articles with 8-70 sentences in their original order:

```json
{'article_id': 'hi-01234', 'sentences': ['...', '...', '...']}
```

At evaluation time, each article is shuffled with a fixed seed, the model scores all pairs, each decoder recovers an order, and tau is computed against the original indices. This measures full-order recovery, not just pair accuracy.

### 17.8 Output schemas

Punctuation JSONL row:

```json
{
  'id': 'hi-00001-0',
  'article_id': 'hi-00001',
  'split': 'train',
  'words': ['भारत', 'एक', 'विशाल', 'देश', 'है'],
  'labels': ['none', 'none', 'none', 'none', 'sentence_end']
}
```

Ordering JSONL row:

```json
{
  'id': 'hi-00001-10',
  'sentence_a': 'पहला वाक्य',
  'sentence_b': 'दूसरा वाक्य',
  'label': 1
}
```

Ordering evaluation row:

```json
{'article_id': 'hi-01234', 'sentences': ['पहला', 'दूसरा', 'तीसरा']}
```

### 17.9 Reproducing data preparation

```bash
python -m src.data_prep --num-articles 3000 --max-train-pairs 40000
```

Smaller first run:

```bash
python -m src.data_prep --num-articles 1200 --max-train-pairs 20000
```

The command writes `data/processed/punctuation/{train,val,test}.jsonl`, `data/processed/ordering/{train,val,test}.jsonl`, `data/processed/ordering/ordering_eval.jsonl`, and two copies of `data_prep_summary.json` (in `data/processed/` and `reports/`).

> **Check yourself.** Why is the same article split used for both stages? *Because the two stages are trained and evaluated on the same documents, making the evaluation comparable and preventing leakage across stages. It also simplifies the story: the test set is a set of unseen articles, period.*

## 18. Stage 1: Punctuation restoration

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Treat punctuation as word-level token classification. Label each word with one of five marks, align the label to the first subword, use a class-weighted loss, and reconstruct sentences from boundary labels. This is the simplest formulation that preserves the input words and fits the memory budget.
</div>

### 18.1 Problem formulation

Input: a list of Hindi words `[w_1, ..., w_n]` with no punctuation. Output: a label `y_i` for each word, where `y_i` is one of:

| Label | Mark | Meaning |
| --- | --- | --- |
| `none` | `''` | no punctuation after this word |
| `comma` | `,` | comma after this word |
| `sentence_end` | `।` | sentence ends here |
| `question` | `?` | question ends here |
| `exclamation` | `!` | exclamation ends here |

The label belongs to the word *before* the mark. For example, in `हुआ।`, the word is `हुआ` and the label is `sentence_end`.

### 18.2 Why token classification

Alternatives:

- **Sequence-to-sequence rewriting:** generate the punctuated text. More flexible, but can change words, is slower, and is harder to evaluate at the word level.
- **Boundary-only binary classification:** predict boundary vs non-boundary, then a second model chooses the mark. Simpler, but loses the joint information and needs two models.
- **Token classification:** one label per word, words preserved, single model. This is the chosen approach.

### 18.3 Subword alignment

The dataset code (`src/punctuation/dataset.py`) does:

```python
encoding = tokenizer(row['words'], is_split_into_words=True,
                     truncation=True, max_length=128)
word_ids = encoding.word_ids()
labels = [-100] * len(encoding['input_ids'])
previous_word = None
for position, word_id in enumerate(word_ids):
    if word_id is None or word_id == previous_word:
        continue
    previous_word = word_id
    labels[position] = LABEL2ID[row['labels'][word_id]]
```

For `अयोध्या` (pieces `अ`, `##य`, `##ोध`, `##्या`), the label goes on `अ`; the three continuation pieces get `-100`. At inference, the code reads the prediction at the first subword of each word.

### 18.4 Model and head

`AutoModelForTokenClassification.from_pretrained(BASE_MODEL, num_labels=5, ...)` loads mBERT and adds a linear head. The head maps each 768-dimensional hidden state to 5 logits. The LoRA adapters are attached to the query and value projections.

### 18.5 Class-weighted loss

`WeightedTrainer` overrides `compute_loss`:

```python
loss_function = torch.nn.CrossEntropyLoss(
    weight=self.class_weight_tensor.to(logits.device),
    ignore_index=-100,
)
loss = loss_function(logits.view(-1, logits.size(-1)), labels.view(-1))
```

This is the only non-standard training component. It is necessary because question and exclamation marks are almost absent.

### 18.6 Inference: sliding windows and triangular weights

`PunctuationRestorer.predict_labels`:

1. Compute windows of 64 words with stride 40.
2. Tokenize each window with `is_split_into_words=True`, `max_length=160`.
3. Run the model in batches of 16.
4. Softmax the logits.
5. For each word, take the **first** subword's probability vector.
6. Weight it by `window_weight(word_id, window_length)`.
7. Accumulate and normalize.
8. Argmax per word.

The triangular weight is the key smoothing trick. Without it, a word at the edge of one window might get a bad prediction that overrides a good prediction from another window.

### 18.7 Rebuilding sentences

`split_into_sentences(words, labels, with_punctuation=True)` walks the words:

- append `word + PUNCTUATION[label]`,
- if the label is a boundary, close the current sentence.

`with_punctuation=False` gives plain sentences for the ordering model, which was trained on plain text (no punctuation). This is important: the ordering model should see the same distribution it was trained on.

### 18.8 What the model learns

The model learns local cues:

- a danda or period after a finite verb,
- a comma after a clause or list item,
- sentence endings after particles like `है`, `था`, `हैं`, `गया`,
- question marks after question words or interrogative particles (rarely, because of data scarcity).

It does not learn global discourse structure; that is the ordering stage's job.

### 18.9 Worked example from the demo notebook

The demo notebook takes the first test window and compares predictions with gold labels:

```text
words in window: 73
अपने               true=none         pred=none
तरह                true=none         pred=none
का                 true=none         pred=none
राज्य-प्रबन्धन     true=none         pred=none
विषयक              true=none         pred=none
यह                 true=none         pred=none
प्राचीनतम          true=none         pred=none
ग्रन्थ             true=none         pred=none
है                 true=sentence_end pred=sentence_end
इसमें              true=none         pred=none
राज्यव्यवस्था      true=comma        pred=comma
कृषि               true=comma        pred=comma
window token accuracy: 0.849
```

This window is handled well: the sentence ending after `है` and the commas after `राज्यव्यवस्था` and `कृषि` are correct. The window accuracy (84.9%) is close to the overall token accuracy (87.2%). Note that `राज्य-प्रबन्धन` is a single word containing a hyphen; the tokenizer splits it into pieces, but the label is still applied to the first piece.

### 18.10 Known limitations

- **Comma F1 is low (0.3161).** Commas are context-dependent and less frequent than sentence endings. The model often misses them.
- **Question/exclamation F1 is 0.** There are only 236 and 153 training examples, and the cap on class weights (20) limits their influence. The model essentially never predicts them.
- **Over-segmentation on dialogue.** Short utterances look like sentence endings. Samples 6 and 7 show this.
- **Truncation.** A 64-word window can exceed 128 subwords; the tail is truncated. This affects long words and complex compounds.
- **Danda vs period.** The model predicts `sentence_end`, which is rendered as `।` regardless of whether the original used `.`. For Hindi output this is appropriate.

> **Check yourself.** Why is the comma the hardest class? *Because comma placement depends on clause structure and style, not just local word identity. A comma can be optional in many positions, so the label is inherently ambiguous. Sentence endings have stronger cues (finite verbs, particles).*

## 19. Stage 2: Sentence ordering

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Learn only pairwise direction, then decode globally. A binary mBERT classifier answers “does A come before B?” for every pair; Borda aggregation turns the probability matrix into a full order. We compare Borda against beam and clustering decoders, and against a TF-IDF-only ablation, rather than assuming one is best.
</div>

### 19.1 Problem formulation

Input: a list of sentences in random order. Output: a permutation that recovers the original narrative order.

The project decomposes this into pairwise decisions: for sentences A and B, predict whether A comes before B. Then a decoder aggregates the pairwise probabilities into a global order.

### 19.2 Why pairwise

Alternatives:

- **Direct permutation prediction:** a model outputs a permutation of `n` items. Hard to train, variable output size, and expensive.
- **Pointer networks:** generate the order one sentence at a time, attending to the remaining sentences. Elegant, but more complex and harder to fit in the budget.
- **Pairwise ranking:** train a binary classifier on pairs, then decode. Simple, generalizes to any `n`, and the pairwise model can be reused by multiple decoders.

The pairwise approach is the standard compromise and matches the course notes.

### 19.3 Building balanced pairs

For consecutive sentences in a Wikipedia article:

```text
(A, B, 1)   # A really comes before B
(B, A, 0)   # the reverse is wrong
```

This gives a 50/50 label distribution, so accuracy is meaningful and no class weighting is needed. The model sees both directions of the same pair, which also teaches symmetry.

### 19.4 Model

`AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)` loads mBERT and adds a linear head on the `[CLS]` vector. The input format is:

```text
[CLS] sentence A [SEP] sentence B [SEP]
```

The tokenizer creates **token type IDs** (segment IDs): 0 for A, 1 for B. BERT's embeddings add a segment embedding, so the model knows which sentence is which. The `[CLS]` representation is passed to the classifier, which outputs two logits.

### 19.5 Inference: the score matrix

`OrderScorer.score_matrix(sentences)`:

1. Enumerate all `i < j` pairs.
2. Batch them through the model.
3. For each pair, take `softmax(logits)[1] = P(A before B)`.
4. Fill `M[i,j] = p`, `M[j,i] = 1 - p`.
5. Diagonal stays 0.5.

The matrix is symmetric-complement, so every decoder can look up either direction.

### 19.6 Decoder 1: Borda

For each sentence `i`:

```text
s_i = (1 / (n - 1)) * sum_{j != i} M[i, j]
```

Sort sentences by `s_i` descending. A sentence that tends to come before others gets a high score and moves to the front.

Why it works: averaging over all pairs reduces the impact of individual pair errors. It is a Borda/Kemeny-style approximation. The true Kemeny order minimizes the number of pairwise disagreements and is NP-hard; Borda is a cheap approximation.

### 19.7 Decoder 2: clusters

1. TF-IDF with `analyzer='char_wb'`, `ngram_range=(2,4)`, `sublinear_tf=True`.
2. KMeans with `k = round(n/6)`, clamped to 1-12.
3. For each cluster, choose the medoid: the member with the highest cosine similarity to the cluster centroid.
4. Order the medoids with a greedy pass over the medoid submatrix.
5. Beam-search inside each cluster.
6. Concatenate cluster orders.

Why cluster? Narrative order often moves between topics. Grouping topically similar sentences reduces the number of expensive pair comparisons and can improve local coherence. In the evaluation, clusters underperform Borda on average, but it is a useful comparison and follows the notes closely.

### 19.8 Decoder 3: beam

Beam search over chains:

- Start from the `beam_width` sentences with the highest earliness score.
- Expand each partial chain by appending a remaining sentence.
- Score the chain by the sum of adjacent pair probabilities.
- Keep the top `beam_width` chains.
- Return the best full chain.

Beam is greedy in the sense that it only scores adjacent transitions, so an early mistake can propagate. It performs worst of the three on average.

### 19.9 TF-IDF ablation

A pure lexical baseline:

- TF-IDF char 2-4 grams,
- cosine similarity matrix,
- diagonal zeroed,
- greedy order by similarity to the current sentence.

This measures how much the mBERT pair model adds over simple lexical cohesion. The answer: a lot, but not enormous. Borda tau 0.2007 vs TF-IDF 0.0631.

### 19.10 Kendall's tau evaluation

For each of 40 held-out articles:

1. Start from the true order `[0, 1, ..., n-1]`.
2. Shuffle with a fixed seed to get `shuffled_indices`.
3. Score the shuffled sentences.
4. Decode to positions in the shuffled list.
5. Map back to original indices.
6. Compute `kendalltau(recovered, truth)`.

One matrix is computed per article and reused by all decoders, so comparing them is cheap.

### 19.11 Worked example from the demo notebook

The demo takes the first test pair:

```text
gold label: 1 (1 = A before B)
P(A before B): 0.711
P(B before A): 0.289
A: यह बाद में ऑगस्ट कॉम्ट(1798-1857 द्वारा 1838 में स्थापित किया गया
B: इससे पहले कॉम्ट ने सामाजिक भौतिकी शब्द का इस्तेमाल किया था लेकिन बाद में वह दूसरों द्वारा अपनाया गया विशेष रूप से बेल्जि
```

The model assigns 0.711 to the correct direction, a clear but not overwhelming signal. The text is noisy: the first sentence is truncated mid-parenthesis, and the second is a long clause. This is typical of real Wikipedia text and explains why ordering is hard.

### 19.12 Known limitations

- **Weak global tau (0.2007).** Pairwise accuracy 65% does not translate into strong full-order recovery.
- **Topical vs narrative order.** The model learns local coherence, but Wikipedia articles often move between topics in ways that are not predictable from adjacent sentences.
- **Long articles.** Quadratic pair count and truncation at 192 tokens. For 99 sentences, the model makes 4,851 calls.
- **No global constraints.** Borda can produce an order that violates transitivity (A before B, B before C, but C before A in the scores). The decoder does not enforce a consistent global ranking.
- **Training data size.** The local adapter saw only 4,000 pairs. The full recipe uses 40,000.

> **Check yourself.** Why can a model with 65% pair accuracy produce an order with tau close to zero? *Because errors are correlated and early mistakes propagate. Global order requires getting many pairs right simultaneously, and a few confidently wrong pairs near topic boundaries can scramble large blocks.*

## 20. Stage 3: Hindi-to-English translation

<div class='approach'>
<strong>APPROACH CHOSEN.</strong> Use a dedicated 200M Indic-to-English encoder-decoder, not a generic multilingual model and not an LLM. Translate sentence by sentence to preserve exact alignment with the Hindi sentences and the groups of five. Use IndicProcessor for language tags and preprocessing.
</div>

### 20.1 Model

`ai4bharat/indictrans2-indic-en-dist-200M` is a 200M-parameter encoder-decoder transformer for Indic-to-English translation. Its configuration (from `models/translation/indictrans2-indic-en-200M/config.json`):

| Property | Value |
| --- | --- |
| Architecture | `IndicTransForConditionalGeneration` |
| Encoder layers | 18 |
| Decoder layers | 18 |
| Model dimension | 512 |
| Attention heads | 8 |
| Feed-forward size | 2,048 |
| Encoder vocabulary | 122,706 |
| Decoder vocabulary | 32,296 |
| Max positions | 256 |
| Dropout | 0.2 |
| Tied embeddings | yes |

The model uses a custom `IndicTrans` architecture with `trust_remote_code=True`.

### 20.2 IndicProcessor

IndicTrans2 expects language tags and preprocessing. `IndicProcessor(inference=True)`:

- adds the source/target tags (`hin_Deva`, `eng_Latn`),
- normalizes and tokenizes text according to the model's conventions,
- postprocesses generated text.

The wrapper calls:

```python
prepared = self.processor.preprocess_batch(batch, src_lang='hin_Deva',
                                           tgt_lang='eng_Latn')
```

Then tokenizes and generates.

> **Subtle bug fixed during development.** The tokenizer is called *without* `src_lang`/`tgt_lang` kwargs because the processor already inserted the tags. Passing them again causes errors. The wrapper also sets `return_attention_mask=True` and `clean_up_tokenization_spaces=True`, and supplies `config.vocab_size` from the decoder vocabulary when the config lacks a top-level `vocab_size`.

### 20.3 Generation

```python
generated = self.model.generate(
    **encoded,
    num_beams=4,
    num_return_sequences=1,
    max_length=256,
    early_stopping=True,
)
```

- **Beam search (4 beams):** explores four partial translations, which usually beats greedy decoding.
- **Max length 256:** enough for the assignment's sentences; longer inputs are truncated.
- **Early stopping:** stop when all beams are finished.

The output is decoded with `skip_special_tokens=True`, then postprocessed by IndicProcessor.

### 20.4 Batching and memory

- Batch size defaults to 4 in the pipeline and can be raised with `--translation-batch-size`.
- On CUDA, dtype is fp16; on MPS/CPU, fp32.
- The model is loaded with `low_cpu_mem_usage=True`.
- After translation, `close()` deletes the model and calls `free_memory()`.

### 20.5 Translation quality

The evaluation on 100 IITB sentences gives BLEU 34.125 and chrF 59.686. Example outputs (from `reports/evaluation.json`) show:

| Hindi | Reference | Hypothesis |
| --- | --- | --- |
| आपकी कार में ब्लैक बॉक्स? | A black box in your car? | A black box in your car? |
| आम तौर पर हाईवे नियोजन जैसा उबाऊ काम भी अचानक गहन बहस तथा जीवंत गठबंधनों का मुद्दा बन गया है। | The usually dull arena of highway planning has suddenly spawned intense debate and colorful alliances. | Even boring work such as highway planning in general has suddenly become an issue of intense debate and lively alliances. |

The first is exact; the second is a paraphrase with different wording but preserved meaning. This is typical of BLEU: exact matches are rewarded heavily, paraphrases less so.

### 20.6 Why not a larger model or an LLM

- The assignment forbids LLMs.
- A 200M model fits the 1.5 GB budget; a 7B model does not.
- IndicTrans2 is specialized for Indic languages and outperforms generic small models on this task.
- A dedicated translation model is easier to evaluate and reproduce than a prompted LLM.

### 20.7 Known limitations

- **Gated access.** The official model requires authentication; the local mirror is a workaround.
- **Sentence-level translation.** Each sentence is translated independently, so cross-sentence discourse is lost.
- **Truncation.** Sentences longer than 256 tokens are cut.
- **Paraphrase.** BLEU penalizes valid paraphrases; chrF is more forgiving.
- **No glossary control.** Domain terms may be translated inconsistently.
- **Local mirror provenance.** The local merged mirror should be replaced with the official checkpoint for a final submission if possible.

> **Check yourself.** Why translate sentence by sentence instead of translating the whole document at once? *Because the assignment requires sentence-aligned Hindi/English outputs and groups of five. Sentence-level translation keeps alignment exact. Document-level translation might improve coherence but would require alignment heuristics and could exceed the model's context.*
