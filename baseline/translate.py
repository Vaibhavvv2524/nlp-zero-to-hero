"""Hindi -> English translation with a small NMT model.

Default model is ``Helsinki-NLP/opus-mt-hi-en`` (MarianMT, ~300 MB), which fits
the 1.5 GB budget. This module was NOT executed on the local machine because
PyTorch/transformers are intentionally not installed here; run it on Colab.

IndicTrans2 note: the distilled Indic->English checkpoint is
``ai4bharat/indictrans2-indic-en-dist-200M`` and normally needs the
``IndicTransToolkit`` package and extra preprocessing. It has NOT been verified
here, so it is not the default.
"""

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-hi-en"


def gpu_memory_report(torch):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"CUDA allocated={allocated:.3f} GB reserved={reserved:.3f} GB"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        return f"MPS allocated={allocated:.3f} GB"
    return "CPU"


class OpusTranslator:
    def __init__(self, model_name=DEFAULT_MODEL, device=None):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device)
        if device == "cuda":
            self.model = self.model.half()
        self.model.eval()

    def memory(self):
        return gpu_memory_report(self.torch)

    def translate(self, texts, batch_size=8, max_length=512, num_beams=4):
        translations = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded, max_length=max_length, num_beams=num_beams
                )
            translations.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return translations

    def close(self):
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
