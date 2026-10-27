"""Compare the trained models side by side.

Reads outputs/<model>/metrics/ for each model, prints a table, and writes
outputs/comparison.json, comparison.csv and comparison.png.

Usage:
    python src/compare_models.py                        # every model in models.py
    python src/compare_models.py --models bert biobert
"""

import argparse
import csv
import json
from pathlib import Path

from models import MODELS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
# Settings that are expected to differ between runs; everything else should match.
PER_MODEL_CONFIG = {"model", "model_id", "output_dir"}
COLUMNS = [
    ("name", "Model", 24, "{}"),
    ("best_epoch", "Best ep", 7, "{:g}"),
    ("val_f1", "Val F1", 7, "{:.4f}"),
    ("test_precision", "Test P", 7, "{:.4f}"),
    ("test_recall", "Test R", 7, "{:.4f}"),
    ("test_f1", "Test F1", 7, "{:.4f}"),
    ("train_minutes", "Train min", 9, "{:.1f}"),
]


def best_epoch(history: list[dict]) -> float | None:
    evals = [e for e in history if "eval_f1" in e]
    return max(evals, key=lambda e: e["eval_f1"])["epoch"] if evals else None


def load_result(key: str, output_dir: Path) -> dict | None:
    metrics_dir = output_dir / key / "metrics"
    try:
        val = json.loads((metrics_dir / "validation.json").read_text())
        test = json.loads((metrics_dir / "test.json").read_text())
        config = json.loads((metrics_dir / "train_config.json").read_text())
        history = json.loads((metrics_dir / "log_history.json").read_text())
    except FileNotFoundError:
        return None
    runtime = next((e["train_runtime"] for e in history if "train_runtime" in e), None)
    return {
        "model": key,
        "name": MODELS[key]["name"],
        "model_id": config["model_id"],
        "best_epoch": best_epoch(history),
        "val_precision": val["validation_precision"],
        "val_recall": val["validation_recall"],
        "val_f1": val["validation_f1"],
        "test_precision": test["test_precision"],
        "test_recall": test["test_recall"],
        "test_f1": test["test_f1"],
        "train_minutes": runtime / 60 if runtime is not None else None,
        "config": config,
        "history": history,
    }


def config_differences(results: list[dict]) -> list[str]:
    """Settings (other than the model) that are not the same across all runs."""
    shared = [{k: v for k, v in r["config"].items() if k not in PER_MODEL_CONFIG} for r in results]
    keys = set().union(*shared) if shared else set()
    return sorted(k for k in keys if len({json.dumps(s.get(k)) for s in shared}) > 1)


def format_table(results: list[dict]) -> str:
    def cell(value, width, fmt):
        return ("-" if value is None else fmt.format(value)).ljust(width)

    lines = ["  ".join(title.ljust(width) for _, title, width, _ in COLUMNS)]
    lines.append("  ".join("-" * width for _, _, width, _ in COLUMNS))
    for r in results:
        lines.append("  ".join(cell(r[key], width, fmt) for key, _, width, fmt in COLUMNS))
    return "\n".join(lines)


def plot(results: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in results:
        evals = [(e["epoch"], e["eval_f1"]) for e in r["history"] if "eval_f1" in e]
        if evals:
            ax_curve.plot(*zip(*evals), marker="o", label=r["name"])
    ax_curve.set(title="Validation entity F1", xlabel="epoch", ylabel="F1")
    ax_curve.legend()

    metrics = [("test_precision", "Precision"), ("test_recall", "Recall"), ("test_f1", "F1")]
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        xs = [m + (i - (len(results) - 1) / 2) * width for m in range(len(metrics))]
        bars = ax_bar.bar(xs, [r[key] for key, _ in metrics], width, label=r["name"])
        ax_bar.bar_label(bars, fmt="%.3f", fontsize=7)
    lowest = min(r[key] for r in results for key, _ in metrics)
    ax_bar.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax_bar.set(title="Test set (best checkpoint)", ylim=(max(0.0, lowest - 0.1), 1.0))
    ax_bar.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    results, missing = [], []
    for key in args.models:
        result = load_result(key, args.output_dir)
        (results if result else missing).append(result or key)

    print("\n" + "=" * 80)
    print("Model comparison: NCBI Disease, entity-level (seqeval), best epoch by validation F1")
    print("=" * 80)
    for key in missing:
        print(f"{key}: no results in {args.output_dir / key / 'metrics'} (not trained, or training failed)")
    if not results:
        return

    print(format_table(results))
    best = max(results, key=lambda r: r["val_f1"])
    print(f"\nHighest validation F1: {best['name']} ({best['val_f1']:.4f}).")
    print("Each model is a single run with one seed, so small test F1 gaps may be run-to-run noise.")
    differing = config_differences(results)
    if differing:
        print(f"WARNING: runs used different settings ({', '.join(differing)}), so they are not directly comparable.")

    rows = [{k: v for k, v in r.items() if k not in ("config", "history")} for r in results]
    (args.output_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
    with (args.output_dir / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(results, args.output_dir / "comparison.png")
    print(f"\nSaved comparison.json, comparison.csv and comparison.png to {args.output_dir}")


if __name__ == "__main__":
    main()
