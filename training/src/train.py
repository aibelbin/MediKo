"""Fine-tune one base model (see models.py) for disease NER on the NCBI Disease corpus.

Trains on `train`, selects the best epoch by validation entity-level F1, then
evaluates once on `test`. Writes the model, tokenizer and metrics to outputs/<model>/.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from models import DEFAULT_MODEL, MODELS, load_tokenizer
from prepare_data import ID2LABEL, IGNORE_INDEX, LABEL2ID, LABELS, load_splits, tokenize_and_align

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def to_label_sequences(predictions, label_ids):
    pred_ids = np.argmax(predictions, axis=-1)
    y_true, y_pred = [], []
    for preds, labels in zip(pred_ids, label_ids):
        keep = labels != IGNORE_INDEX
        y_true.append([ID2LABEL[int(l)] for l in labels[keep]])
        y_pred.append([ID2LABEL[int(p)] for p in preds[keep]])
    return y_true, y_pred


def compute_metrics(eval_pred):
    y_true, y_pred = to_label_sequences(*eval_pred)
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL, help="base model to fine-tune")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1, help="cap steps (for a quick smoke test)")
    parser.add_argument("--limit", type=int, default=0, help="use only N examples per split (smoke test)")
    parser.add_argument("--output-dir", type=Path, help="default: outputs/<model>")
    args = parser.parse_args()
    args.output_dir = args.output_dir or OUTPUT_DIR / args.model
    model_id = MODELS[args.model]["id"]

    set_seed(args.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, cuda={torch.cuda.is_available()}")
    print(f"model: {args.model} ({model_id})")

    tokenizer = load_tokenizer(args.model)
    raw = load_splits()
    if args.limit:
        for split in raw:
            raw[split] = raw[split].select(range(min(args.limit, len(raw[split]))))
    for split, ds in raw.items():
        lengths = [len(tokenizer(t, is_split_into_words=True)["input_ids"]) for t in ds["tokens"]]
        truncated = sum(n > args.max_length for n in lengths)
        print(f"{split}: {len(ds)} examples, max subword length {max(lengths)}, "
              f"truncated at {args.max_length}: {truncated}")

    tokenized = raw.map(
        lambda b: tokenize_and_align(b, tokenizer, args.max_length),
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    checkpoints = args.output_dir / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=0.1,  # float in [0, 1) = ratio of total steps
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=args.max_steps if args.max_steps > 0 else None,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=args.max_steps if args.max_steps > 0 else None,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    val_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    print("\nBest-checkpoint validation metrics:", json.dumps(val_metrics, indent=2))

    # Held-out test set: evaluated once, after model selection.
    test_out = trainer.predict(tokenized["test"], metric_key_prefix="test")
    y_true, y_pred = to_label_sequences(test_out.predictions, test_out.label_ids)
    report = classification_report(y_true, y_pred, digits=4)
    print("\nTest metrics:", json.dumps(test_out.metrics, indent=2))
    print("\nTest classification report:\n" + report)

    (metrics_dir / "validation.json").write_text(json.dumps(val_metrics, indent=2))
    (metrics_dir / "test.json").write_text(json.dumps(test_out.metrics, indent=2))
    (metrics_dir / "test_classification_report.txt").write_text(report)
    (metrics_dir / "train_config.json").write_text(
        json.dumps({**vars(args), "output_dir": str(args.output_dir), "model_id": model_id}, indent=2)
    )
    (metrics_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    final_dir = args.output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved best model + tokenizer to {final_dir}")
    print(f"Saved metrics to {metrics_dir}")


if __name__ == "__main__":
    main()
