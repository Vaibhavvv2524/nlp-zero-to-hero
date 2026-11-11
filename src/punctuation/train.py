"""Fine-tune mBERT for punctuation restoration with LoRA.

The base model is ``bert-base-multilingual-cased`` (about 178M parameters).
Only small LoRA matrices and the classification head are trained, which keeps
GPU memory comfortably below the 1.5 GB assignment limit and makes training
finish in under an hour on a free Colab T4.

Usage
-----
    python -m src.punctuation.train \
        --punct-data-dir data/processed/punctuation \
        --output-dir models/punctuation
"""

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ..constants import (
    BASE_MODEL,
    BOUNDARY_IDS,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    MEMORY_LIMIT_GB,
    REPORTS,
    PUNCT_DATA,
    PUNCT_MODEL_DIR,
    SEED,
)
from ..utils import (
    attention_implementation,
    ensure_dir,
    free_memory,
    gpu_memory,
    limit_gpu_memory,
    read_jsonl,
    save_json,
    set_seed,
)
from .dataset import PunctuationDataset


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    boundary_true = np.isin(labels, BOUNDARY_IDS)
    boundary_pred = np.isin(predictions, BOUNDARY_IDS)
    precision, recall, f1, _ = precision_recall_fscore_support(
        boundary_true, boundary_pred, average="binary", zero_division=0
    )
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(f1),
    }
    for index, name in enumerate(LABELS):
        metrics[f"f1_{name}"] = float(per_class_f1[index])
    return metrics


def class_weights(path):
    """Inverse-frequency weights, with rare labels capped for stability."""
    counts = np.zeros(len(LABELS), dtype=np.float64)
    for row in read_jsonl(path):
        for label in row["labels"]:
            counts[LABEL2ID[label]] += 1
    total = counts.sum()
    weights = np.zeros(len(LABELS), dtype=np.float64)
    for index, count in enumerate(counts):
        if count > 0:
            weights[index] = total / (len(LABELS) * count)
    weights = np.clip(weights, 0.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32), counts


class WeightedTrainer(Trainer):
    """Trainer that uses class weights, needed for rare question/exclamation marks."""

    def __init__(self, class_weight_tensor=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weight_tensor = class_weight_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_function = torch.nn.CrossEntropyLoss(
            weight=self.class_weight_tensor.to(logits.device),
            ignore_index=-100,
        )
        loss = loss_function(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def build_training_arguments(
    output_dir, epochs, batch_size, learning_rate, train_size
):
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = (
        "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    )
    scheduler_kwargs = {}
    if "warmup_ratio" in parameters:
        scheduler_kwargs["warmup_ratio"] = 0.06
    else:
        steps_per_epoch = max(1, -(-train_size // (batch_size * 2)))
        scheduler_kwargs["warmup_steps"] = max(
            1, int(0.06 * steps_per_epoch * epochs)
        )
    return TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        **scheduler_kwargs,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="boundary_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
        **{strategy_name: "epoch"},
    )


def build_trainer(model, arguments, tokenizer, train_dataset, val_dataset, weights):
    common = {
        "model": model,
        "args": arguments,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorForTokenClassification(tokenizer),
        "compute_metrics": compute_metrics,
        "class_weight_tensor": weights,
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        return WeightedTrainer(processing_class=tokenizer, **common)
    return WeightedTrainer(tokenizer=tokenizer, **common)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--punct-data-dir", type=Path, default=PUNCT_DATA)
    parser.add_argument("--output-dir", type=Path, default=PUNCT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--memory-gb", type=float, default=MEMORY_LIMIT_GB)
    args = parser.parse_args(argv)

    set_seed(SEED)
    memory_report = limit_gpu_memory(args.memory_gb)
    print("Memory limit:", memory_report)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = PunctuationDataset(
        args.punct_data_dir / "train.jsonl", tokenizer
    )
    val_dataset = PunctuationDataset(args.punct_data_dir / "val.jsonl", tokenizer)
    test_dataset = PunctuationDataset(args.punct_data_dir / "test.jsonl", tokenizer)

    if args.max_train_samples is not None:
        train_dataset.rows = train_dataset.rows[: args.max_train_samples]
        print("Using", len(train_dataset), "training windows")

    weights, counts = class_weights(args.punct_data_dir / "train.jsonl")
    print("Label counts:", dict(zip(LABELS, counts.astype(int).tolist())))
    print("Class weights:", [round(float(value), 3) for value in weights])

    attention = attention_implementation()
    base_model_kwargs = {"attn_implementation": attention} if attention else {}
    base_model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        **base_model_kwargs,
    )
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print("Memory after model load:", gpu_memory())

    arguments = build_training_arguments(
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        len(train_dataset),
    )
    trainer = build_trainer(
        model,
        arguments,
        tokenizer,
        train_dataset,
        val_dataset,
        weights,
    )

    trainer.train()
    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(
        eval_dataset=test_dataset, metric_key_prefix="test"
    )
    print("Validation:", val_metrics)
    print("Test:", test_metrics)

    adapter_dir = ensure_dir(args.output_dir / "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    summary = {
        "base_model": BASE_MODEL,
        "adapter_dir": str(adapter_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_windows": len(train_dataset),
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "label_counts": dict(zip(LABELS, counts.astype(int).tolist())),
        "class_weights": [float(value) for value in weights],
        "memory_limit": memory_report,
        "memory_at_end": gpu_memory(),
    }
    save_json(args.output_dir / "training_summary.json", summary)
    save_json(REPORTS / "punctuation_training.json", summary)
    print("Saved adapter to", adapter_dir)

    del trainer, model, base_model
    free_memory()
    return test_metrics


if __name__ == "__main__":
    main()
