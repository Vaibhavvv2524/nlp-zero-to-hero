# 8. Stage 3: translation with IndicTrans2

<div class="decision">

**APPROACH CHOSEN.** Translate each ordered Hindi sentence with `ai4bharat/indictrans2-indic-en-dist-200M`, using IndicTransToolkit for preprocessing and postprocessing, beam search with four beams, and batched inference. There is exactly one translation model and no fallback. Translation runs only after punctuation and ordering, so it receives well-formed, ordered Hindi sentences.

</div>

## 8.1 Where translation sits in the argument

Translation is the only generative stage in the pipeline. That makes it powerful and risky:

- It can produce fluent English that the earlier stages could never produce.
- It can also introduce errors that are invisible to word-level metrics.
- It is the stage where hallucination is possible, even though the system as a whole is not an LLM pipeline.

The design response is to isolate translation behind a narrow interface: a list of Hindi sentences goes in, a list of English sentences comes out, one output per input. The stage cannot change the number of sentences, the order, or the Hindi text used by the rest of the system.

## 8.2 Why IndicTrans2

| Candidate | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| **IndicTrans2 200M distilled** | Indic-specific; strong Hindi–English quality; small enough for the budget; official toolkit | Extra preprocessing dependency; gated upstream; one direction only | **Chosen** |
| NLLB-200 | Multilingual; widely used | Larger; less Indic-specialised at this size; heavier memory | Rejected |
| mBART / mT5 | Available in transformers | Weaker Hindi–English quality; needs fine-tuning | Rejected |
| Helsinki-NLP OPUS-MT | Very small; easy | Lower quality; limited context | Rejected |
| LLM translation | Fluent | Forbidden; expensive; non-deterministic; opaque | Rejected |
| Dictionary / transliteration | No model | Not translation; fails on grammar | Rejected |

IndicTrans2 is trained specifically for Indic→Indic and Indic→English translation. The 200M distilled variant is the smallest official checkpoint that still gives usable quality, which matters for the 1.5GB memory target.

The exact model id is:

```text
ai4bharat/indictrans2-indic-en-dist-200M
```

The local checkout uses a mirrored copy at `models/translation/indictrans2-indic-en-200M` because the upstream repository is gated and this machine has no Hugging Face token. The mirror is the same checkpoint; it is not a different model.

## 8.3 Architecture and interface

<figure class="diagram">
<img src="approach_assets/07_translation.svg" alt="IndicTrans2 translation stage">
<figcaption><strong>Figure 8.1.</strong> Sentences are tagged with source and target language codes, tokenized, translated by the encoder–decoder model with beam search, decoded, and postprocessed. Empty inputs stay empty so the output list aligns with the input list.</figcaption>
</figure>

IndicTrans2 is an encoder–decoder transformer. The distilled 200M checkpoint is roughly 0.4GB in fp16. The pipeline uses the Hugging Face `AutoTokenizer` and `AutoModelForSeq2SeqLM` with `trust_remote_code=True`, plus `IndicProcessor` from IndicTransToolkit.

| Setting | Value |
| --- | --- |
| Source language | `hin_Deva` |
| Target language | `eng_Latn` |
| Parameters | about 200M |
| CUDA dtype | fp16 |
| MPS dtype | fp32 |
| Batch size | 4 |
| Beams | 4 |
| Max generation length | 256 |
| Early stopping | true |
| Return sequences | 1 |

### 8.3.1 Preprocessing

`IndicProcessor.preprocess_batch` does the language-tagging and normalisation that IndicTrans2 expects. The translator does not hand-build prompt strings. This matters: the model was trained with the toolkit's conventions, and bypassing them changes the input distribution.

The tokenizer then truncates to the model maximum, pads to the longest item in the batch, and returns attention masks.

### 8.3.2 Generation

```python
generated = model.generate(
    **encoded,
    num_beams=4,
    num_return_sequences=1,
    max_length=256,
    early_stopping=True,
)
```

Beam search is used because translation quality matters more than latency. Four beams keep the cost moderate; larger beam counts gave diminishing returns for this model size.

### 8.3.3 Postprocessing

`IndicProcessor.postprocess_batch` removes special tags, fixes spacing, and applies the toolkit's target-language cleanup. The result is a plain English string per input sentence.

### 8.3.4 Alignment

The pipeline flattens all ordered Hindi sentences from all input files into one list:

```text
flat_sentences = [all sentences from sample-1, then sample-2, ...]
translations = translator.translate(flat_sentences, batch_size=4)
```

It then walks the records again with an offset and slices the translation list back per file. This gives one English sentence per Hindi sentence and preserves order. Empty strings are preserved as empty strings, so indices never shift.

## 8.4 Why translate sentence by sentence

| Option | Pros | Cons |
| --- | --- | --- |
| **One sentence per call (batched)** | Alignment is exact; memory is predictable; one bad sentence cannot corrupt a document | Loses cross-sentence context; pronouns can be ambiguous |
| Whole document | More context | Long inputs; alignment is harder; memory spikes; one failure affects everything |
| Paragraph chunks | Some context | Still needs alignment; chunk boundaries can split discourse |
| Translate before ordering | Translation could help order | Ordering English changes the task; errors compound differently |

Sentence-level translation is the right compromise for this pipeline because stage 1 already defines sentence boundaries. The output format also requires sentence-level English lines. Cross-sentence context is a documented limitation, not an accident.

## 8.5 Measured quality

The translation stage is evaluated on 100 examples from `cfilt/iitb-english-hindi`:

| Metric | Value |
| --- | ---: |
| Corpus BLEU | 34.125 |
| chrF (word order 2) | 59.686 |
| Examples | 100 |

### 8.5.1 What the metrics mean

- **BLEU** measures n-gram precision against reference translations with a brevity penalty. It rewards exact phrasing and penalises omissions and insertions. A score of 34 is a solid but clearly imperfect translation result.
- **chrF** measures character n-gram overlap. It is more forgiving of morphological variation and word-order differences, which is why it is higher.

Neither metric measures factual correctness, fluency, or discourse coherence. Both are corpus-level scores on a benchmark sample, not on the assignment files.

### 8.5.2 Sample behaviour

From `reports/evaluation.json`:

| Hindi input (abridged) | Reference | Hypothesis |
| --- | --- | --- |
| `आपकी कार में ब्लैक बॉक्स?` | A black box in your car? | A black box in your car? |
| `जबकि अमेरिका के सड़क योजनाकार ...` | As America's road planners struggle ... | While U.S. road planners are struggling ... |
| `यह डिवाइस, जो मोटर-चालक ...` | The devices, which track every mile ... | The device, which tracks every mile ... |
| `आम तौर पर हाईवे नियोजन ...` | The usually dull arena ... | Even boring work such as highway planning ... |

The first example is exact. The others are fluent paraphrases with small meaning shifts: singular/plural, added framing, and changed emphasis. This is the expected behaviour of a strong but small translation model.

## 8.6 Timing and memory

From `reports/run_summary.json`, all seven assignment files:

| Stage | Time |
| --- | ---: |
| Punctuation | 2.31 s |
| Ordering | 130.53 s |
| Translation | 102.17 s |

The seven files contain 443 sentences after punctuation and ordering. Translation is therefore about 0.23 seconds per sentence on MPS with batch size 4 and four beams. Ordering is the slowest stage because of the quadratic pair count.

The translator is loaded only after the ordering model is closed. Models are never held simultaneously, which is how the pipeline respects the memory cap.

## 8.7 Pros of this stage design

1. **Strong Indic-specific quality.** The model is built for Hindi→English.
2. **Small footprint.** The 200M distilled checkpoint fits the budget.
3. **Exact alignment.** One output per input sentence, order preserved.
4. **No fallback ambiguity.** A single model path is easy to audit.
5. **Toolkit compatibility.** Preprocessing follows the model's training convention.
6. **Beam search quality.** Four beams improve output without excessive cost.
7. **Clean interface.** Translation cannot reorder or merge sentences.

## 8.8 Cons and failure modes

| Weakness | Cause | Effect |
| --- | --- | --- |
| **Meaning drift** | Small model; beam search picks fluent but approximate output. | Paraphrases can change emphasis or number. |
| **No cross-sentence context** | Sentences are translated independently. | Pronouns and discourse links can be wrong. |
| **Named entities** | Transliteration and spelling vary. | Names and places may be inconsistent. |
| **Long-sentence truncation** | 256-token generation limit. | Very long sentences can lose their ending. |
| **Domain mismatch** | IITB benchmark is news-like; assignment samples differ. | Benchmark BLEU may not reflect assignment quality. |
| **Metric blind spots** | BLEU/chrF do not measure facts or fluency. | A high score can hide meaning errors. |
| **Gated upstream model** | No HF token on this machine. | Requires the local mirror. |
| **No glossary** | No term dictionary or constrained decoding. | Technical terms can drift. |
| **MPS is fp32** | fp16 is CUDA-only here. | Slower and more memory than fp16. |

<div class="warning">

**Do not claim that BLEU 34.125 measures the assignment output.** It measures 100 IITB sentence pairs. The assignment files have no reference English translations, so their translation quality is only inspected qualitatively.

</div>

## 8.9 Alternative choices considered

| Alternative | Why rejected |
| --- | --- |
| Translate before ordering | Ordering English sentences changes the learned task and compounds errors differently. |
| Translate whole documents | Alignment and memory become harder; the required output is sentence-level. |
| Use a larger IndicTrans2 checkpoint | Better quality, but does not fit the 1.5GB target. |
| Use NLLB-200 | Larger and less Indic-specialised at the size that fits. |
| Use an LLM | Forbidden and non-deterministic. |
| Add a second translation model as fallback | Doubles memory and complicates the claim; no fallback is simpler and auditable. |

A stronger future design would translate paragraph groups with a glossary and a consistency pass. That is an improvement, not the shipped approach.

## 8.10 Implementation map

| Function / class | File | Role |
| --- | --- | --- |
| `_import_indic_processor` | `src/translation/indictrans.py` | compatibility shim for transformers 4.x/5.x |
| `IndicTrans2Translator.__init__` | `src/translation/indictrans.py` | loads tokenizer, processor, model, dtype |
| `IndicTrans2Translator.translate` | `src/translation/indictrans.py` | batches, preprocesses, generates, postprocesses, aligns |
| `evaluate_translation` | `src/evaluate.py` | BLEU/chrF on IITB samples |
| `run` | `src/pipeline.py` | flattens sentences, translates, splits back per file |

<div class="pagebreak"></div>
