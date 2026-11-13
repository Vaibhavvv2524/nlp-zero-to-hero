"""Hindi -> English translation with IndicTrans2.

Model: ``ai4bharat/indictrans2-indic-en-dist-200M`` (about 200M parameters,
roughly 0.4 GB in fp16). This is the only translation model used by the final
pipeline - there is no fallback.

IndicTrans2 expects the ``IndicTransToolkit`` preprocessing package:

    pip install IndicTransToolkit

Usage
-----
    from src.translation.indictrans import IndicTrans2Translator

    translator = IndicTrans2Translator()
    print(translator.translate(["भारत एक विशाल देश है।"]))
    translator.close()
"""

from pathlib import Path

from ..constants import (
    MEMORY_LIMIT_GB,
    SRC_LANG,
    TGT_LANG,
    TRANSLATION_MODEL,
)
from ..utils import free_memory, get_device, gpu_memory, limit_gpu_memory

_PRECOMPOSED_NUKTA = {
    "\u0958": "\u0915\u093c",
    "\u0959": "\u0916\u093c",
    "\u095a": "\u0917\u093c",
    "\u095b": "\u091c\u093c",
    "\u095c": "\u0921\u093c",
    "\u095d": "\u0922\u093c",
    "\u095e": "\u092b\u093c",
    "\u095f": "\u092f\u093c",
}


def _decompose_nukta(text):
    """Split precomposed nukta letters into base letter + nukta sign.

    IndicTrans2 otherwise decodes these letters as literal ``\\ u093C``
    artifacts in the English output.
    """
    return "".join(_PRECOMPOSED_NUKTA.get(char, char) for char in text)


def _import_indic_processor():
    """Import ``IndicProcessor`` across the transformers 4.x / 5.x split.

    ``IndicTransToolkit`` still imports ``PreTrainedTokenizerBase`` from
    ``transformers.tokenization_utils``, a path that transformers 5 removed.
    Re-export the class from its new module before retrying the import.
    """
    try:
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        import transformers.tokenization_utils as tokenization_utils
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
        from IndicTransToolkit import IndicProcessor
    return IndicProcessor


class IndicTrans2Translator:
    """Batch translator around IndicTrans2."""

    def __init__(
        self,
        model_name=TRANSLATION_MODEL,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        device=None,
        memory_gb=MEMORY_LIMIT_GB,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            IndicProcessor = _import_indic_processor()
        except ImportError as error:
            raise ImportError(
                "IndicTransToolkit is required for IndicTrans2 preprocessing. "
                "Run: bash scripts/colab_setup.sh"
            ) from error

        self.torch = torch
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.memory_report = limit_gpu_memory(memory_gb)
        self.device = device or get_device()
        self.processor = IndicProcessor(inference=True)

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        config = self.model.config
        if getattr(config, "vocab_size", None) is None:
            config.vocab_size = getattr(
                config,
                "decoder_vocab_size",
                getattr(config, "encoder_vocab_size", None),
            )
        self.model.to(self.device)
        self.model.eval()

    def translate(self, texts, batch_size=4, num_beams=4, max_length=256):
        """Translate a list of Hindi strings, preserving order and empty items."""
        import torch

        texts = [_decompose_nukta(text) for text in texts]
        # Keep empty strings as empty outputs so the result list still lines up
        # with the caller's sentence indices.
        indexed = [
            (index, text)
            for index, text in enumerate(texts)
            if text and text.strip()
        ]
        results = [""] * len(texts)
        if not indexed:
            return results

        for start in range(0, len(indexed), batch_size):
            chunk = indexed[start : start + batch_size]
            batch = [text for _, text in chunk]
            # The toolkit adds IndicTrans2 language tags and normalises text.
            prepared = self.processor.preprocess_batch(
                batch, src_lang=self.src_lang, tgt_lang=self.tgt_lang
            )
            encoded = self.tokenizer(
                prepared,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            )
            encoded = {
                key: value.to(self.device) for key, value in encoded.items()
            }
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    num_beams=num_beams,
                    num_return_sequences=1,
                    max_length=max_length,
                    early_stopping=True,
                )
            decoded = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            # Postprocessing removes tags and fixes spacing/tokenisation.
            translations = self.processor.postprocess_batch(
                decoded, lang=self.tgt_lang
            )
            for (index, _), translation in zip(chunk, translations):
                results[index] = translation
        return results

    def memory(self):
        return gpu_memory()

    def close(self):
        del self.model
        self.model = None
        free_memory()
