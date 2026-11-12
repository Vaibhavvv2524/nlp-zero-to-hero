# Hindi Text Restoration and Translation

A beginner-friendly NLP project that takes Hindi text with missing punctuation and shuffled sentences, then:

1. puts punctuation and sentence boundaries back,
2. puts the sentences in a sensible order,
3. translates them into English,
4. keeps every five sentences together as a group.

**No LLM is used.** The project uses a multilingual language model for punctuation and ordering, and IndicTrans2 for translation.

**Repository:** https://github.com/Vaibhavvv2524/nlp-zero-to-hero

**Full Colab build:** https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/colab_end_to_end.ipynb

**Executed demo:** https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/pipeline_demo.ipynb

---

## What it does, in plain words

Imagine three helpers working on the same Hindi article:

1. **The editor** decides where commas and full stops belong.
2. **The librarian** decides which sentence probably comes first, second, third, and so on.
3. **The translator** turns each Hindi sentence into English without changing the order.

The project connects those three helpers into one pipeline.

```text
Input:  Hindi words, no punctuation, sentences shuffled
        भारत एक विशाल देश है यह एक परीक्षण है

Stage 1: add punctuation and sentence boundaries
        भारत एक विशाल देश है। यह एक परीक्षण है।

Stage 2: put the sentences in a likely order

Stage 3: translate to English and group every five sentences

Output: Hindi sentences, Hindi groups, English sentences, English groups
```

## The three stages

| Stage | Simple job | Model | Main result |
| --- | --- | --- | --- |
| **Punctuation** | Decide what follows each word | mBERT + LoRA | Boundary F1 **0.8842** |
| **Ordering** | Decide which sentence comes first | mBERT + LoRA + Borda | Pair accuracy **0.6543**, Kendall tau **0.2007** |
| **Translation** | Turn Hindi sentences into English | IndicTrans2-200M | BLEU **34.125**, chrF **59.686** |

These numbers come from a smaller local training run, not the full training recipe. The README explains this honestly later.

---

## Notebooks

All five notebooks are in the `notebooks/` folder and are also available through GitHub and Google Colab.

| # | Notebook | What it is | Best for |
| ---: | --- | --- | --- |
| 1 | [`pipeline_demo.ipynb`](https://github.com/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/pipeline_demo.ipynb) · [Colab](https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/pipeline_demo.ipynb) | An already-executed demo with saved outputs and scores | Seeing the final result quickly |
| 2 | [`colab_end_to_end.ipynb`](https://github.com/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/colab_end_to_end.ipynb) · [Colab](https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/colab_end_to_end.ipynb) | The full project: clone, install, prepare data, train, evaluate, run the pipeline | Running the project from scratch |
| 3 | [`lesson_01_basics.ipynb`](https://github.com/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/lesson_01_basics.ipynb) · [Colab](https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/lesson_01_basics.ipynb) | A gentle check of Python, NumPy, and pandas | Absolute beginners |
| 4 | [`lesson_02_tokenization.ipynb`](https://github.com/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/lesson_02_tokenization.ipynb) · [Colab](https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/lesson_02_tokenization.ipynb) | What tokens and sentences are, with small examples | Learning the basic NLP idea |
| 5 | [`lesson_03_preprocessing.ipynb`](https://github.com/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/lesson_03_preprocessing.ipynb) · [Colab](https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/lesson_03_preprocessing.ipynb) | Cleaning, tokenization, stopwords, stemming, and lemmatization | Understanding text preparation |

> **Important:** the demo notebook is read-only evidence. It loads local model files that are not committed to GitHub, so **do not click Run all** there. Use the full Colab build if you want to run the project.

The three lesson notebooks are for learning. They are not part of the final Hindi pipeline.

---

## Quick start

### I only want to see the results

1. Open the [executed demo](https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/pipeline_demo.ipynb).
2. Read the saved outputs. Do not run the cells.
3. For the final files, look in [`outputs/`](outputs/).

### I want to run the full project

1. Open the [full Colab build](https://colab.research.google.com/github/Vaibhavvv2524/nlp-zero-to-hero/blob/main/notebooks/colab_end_to_end.ipynb).
2. In Colab, choose **Runtime > Change runtime type > T4 GPU**.
3. Press **Runtime > Run all**.
4. The first code cell clones this repository automatically. You do not need to upload a folder.
5. Optional: for translation, accept the IndicTrans2 licence on Hugging Face and add a Colab secret named `HF_TOKEN`.
6. The notebook installs dependencies, prepares data, trains the models, evaluates them, runs the pipeline, and checks the outputs.

The full run takes roughly **30–60 minutes** on a free Colab T4. Training works without `HF_TOKEN`; translation needs it because the translation model is gated.

---

## What you get

For every input file, the pipeline writes four aligned files:

| File | What it contains |
| --- | --- |
| `sample-N.sentences_hi.txt` | One Hindi sentence per line, in the recovered order |
| `sample-N.groups_hi.txt` | The same Hindi sentences, five per line |
| `sample-N.sentences_en.txt` | One English translation per line |
| `sample-N.groups_en.txt` | The same English translations, five per line |

Line *i* in the Hindi sentence file matches line *i* in the English sentence file.

Reports are written to `reports/`:

- `data_prep_summary.json` — how much training data was created
- `evaluation.json` — scores for all three stages
- `run_summary.json` — sentence counts, groups, and timings

---

## Results to remember

| Stage | Metric | Value |
| --- | --- | ---: |
| Punctuation | Boundary F1 | **0.8842** |
| Punctuation | Comma F1 | 0.3161 |
| Punctuation | Question / exclamation F1 | 0.0 / 0.0 |
| Ordering | Pair accuracy | 0.6543 |
| Ordering | Borda Kendall tau | 0.2007 |
| Translation | BLEU | 34.125 |
| Translation | chrF | 59.686 |

**How to read this honestly:**

- Punctuation works well for sentence endings, but commas and rare marks are weak.
- Ordering is the weakest stage. It beats a simple baseline but does not fully recover the original order.
- Translation is usable, but BLEU and chrF are measured on 100 IITB benchmark pairs, not on the assignment files.

The shipped adapters were trained on a smaller subset. The full training recipe uses more examples.

---

## How it works, simply

### Stage 1: punctuation

The model reads words and chooses one label for each word:

| Label | Meaning |
| --- | --- |
| `none` | Nothing after this word |
| `comma` | Add a comma |
| `sentence_end` | Add a danda `।` and end the sentence |
| `question` | Add a question mark |
| `exclamation` | Add an exclamation mark |

The model does **not** rewrite words. It only labels them.

### Stage 2: ordering

The model reads two sentences and answers one question: **does A come before B?**

It is trained on consecutive Wikipedia sentences. Each real pair `(A, B)` is used as a positive example, and the swapped pair `(B, A)` is used as a negative example. After scoring every pair, Borda decoding turns the scores into one order.

### Stage 3: translation

Each ordered Hindi sentence is translated separately with IndicTrans2. Translating one sentence at a time keeps the Hindi and English lines aligned.

No fallback translator is used. If the translation model is missing, the pipeline stops with a clear error.

---

## Training data

The models are trained on **Hindi Wikipedia**, not on the seven sample files.

The data preparation does this:

1. Stream Hindi Wikipedia articles.
2. Split each article into sentences using its real punctuation.
3. Remove the punctuation from the words but keep it as the label.
4. Group sentences into windows and shuffle them, matching the assignment input.
5. Build ordering pairs from consecutive sentences and their reversed versions.
6. Split by article so the same article never appears in both training and testing.

The sample files are only used for prediction, not training.

---

## What is in the repository

```text
notebooks/          five notebooks: two project notebooks, three lessons
src/                the pipeline code
  punctuation/      Stage 1
  ordering/         Stage 2
  translation/      Stage 3
  pipeline.py       connects all three stages
  evaluate.py       computes the scores
data/raw/           the seven sample input files
outputs/            final Hindi and English outputs
reports/            JSON metrics and run summaries
scripts/            setup and verification scripts
docs/               longer guides and design notes
```

The generated `data/processed/` files and the trained `models/` are intentionally not committed. They are created by the full run.

---

## Run on your own computer

Install the local requirements:

```bash
python -m pip install -r requirements.txt
```

Then run the pipeline:

```bash
python -m src.pipeline --strategy borda
```

A fast check without translation:

```bash
python -m src.pipeline --limit 1 --no-translate
```

Training commands are in the full Colab notebook and in the longer study guide.

---

## Honest limits

- **Ordering is not solved.** It improves the order but can still make mistakes.
- **Commas are hard.** The comma score is much lower than the sentence-ending score.
- **Questions and exclamations are rare in the training data.** The model currently scores zero on them.
- **Translation is sentence-level.** It loses some cross-sentence context.
- **The local scores are subset scores.** Do not describe them as full-data results.
- **The 1.5GB CUDA memory cap is implemented, but it was not measured on a CUDA machine here.**

---

## Troubleshooting

| Problem | What to try |
| --- | --- |
| `IndicTransToolkit` import or build error | Run `bash scripts/colab_setup.sh` |
| `peft` / `transformers` mismatch | Install `requirements-colab.txt`; do not upgrade to transformers 5 |
| Translation says `401` or gated repo | Accept the IndicTrans2 licence and add `HF_TOKEN` |
| CUDA out of memory | Lower `--batch-size`, keep `--memory-gb 1.5`, restart the runtime |
| Notebook clone fails | Check the internet connection, then rerun the setup cell |
| Self-check fails | Read the message from `scripts/verify_colab_run.py`; it names the missing file or metric |

---

## Learn more

The short guide is in the repository as Markdown and HTML:

- [`docs/learning/00_learning_plan.md`](docs/learning/00_learning_plan.md)
- [`docs/what-to-learn-priority-order.html`](docs/what-to-learn-priority-order.html)

Longer PDFs are generated locally and are not committed:

- `docs/assblasternlp.pdf` — beginner study guide
- `docs/holy-approach.pdf` — detailed design dossier

The short answer: **start with the notebooks, then read the short guide, and only open the long PDFs if you want more depth.**
