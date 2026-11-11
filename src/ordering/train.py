"""Fine-tune mBERT to decide whether sentence A comes before sentence B.

This is the pairwise scorer from the notes: a binary classifier over the pair
``[CLS] A [SEP] B [SEP]``. At inference time the probabilities for all sentence
pairs are combined with clustering and beam search into a full ordering.

Usage
-----
    python -m src.ordering.train \
        --order-data-dir data/processed/ordering \
        --output-dir models/ordering
"""

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..constants import (
    BASE_MODEL,
    MEMORY_LIMIT_GB,
    REPORTS,
    ORDER_DATA,
    ORDER_MODEL_DIR,
    SEED,
)
from ..utils import (
    attention_implementation,
    ensure_dir,
    free_memory,
    gpu_memory,
    limit_gpu_memory,
    save_json,
    set_seed,
)
from .pairs import SentencePairDataset


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


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
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
        **{strategy_name: "epoch"},
    )


def build_trainer(model, arguments, tokenizer, train_dataset, val_dataset):
    common = {
        "model": model,
        "args": arguments,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        return Trainer(processing_class=tokenizer, **common)
    return Trainer(tokenizer=tokenizer, **common)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-data-dir", type=Path, default=ORDER_DATA)
    parser.add_argument("--output-dir", type=Path, default=ORDER_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--memory-gb", type=float, default=MEMORY_LIMIT_GB)
    args = parser.parse_args(argv)

    set_seed(SEED)
    memory_report = limit_gpu_memory(args.memory_gb)
    print("Memory limit:", memory_report)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = SentencePairDataset(
        args.order_data_dir / "train.jsonl", tokenizer
    )
    val_dataset = SentencePairDataset(args.order_data_dir / "val.jsonl", tokenizer)
    test_dataset = SentencePairDataset(
        args.order_data_dir / "test.jsonl", tokenizer
    )

    if args.max_train_samples is not None:
        train_dataset.rows = train_dataset.rows[: args.max_train_samples]
        print("Using", len(train_dataset), "training pairs")

    attention = attention_implementation()
    base_model_kwargs = {"attn_implementation": attention} if attention else {}
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "wrong_order", 1: "correct_order"},
        label2id={"wrong_order": 0, "correct_order": 1},
        **base_model_kwargs,
    )
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
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
        model, arguments, tokenizer, train_dataset, val_dataset
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
        "train_pairs": len(train_dataset),
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "memory_limit": memory_report,
        "memory_at_end": gpu_memory(),
    }
    save_json(args.output_dir / "training_summary.json", summary)
    save_json(REPORTS / "ordering_training.json", summary)
    print("Saved adapter to", adapter_dir)

    del trainer, model, base_model
    free_memory()
    return test_metrics


if __name__ == "__main__":
    main()
