"""Plot training loss and validation F1 from outputs/metrics/log_history.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"


def main() -> None:
    history = json.loads((METRICS_DIR / "log_history.json").read_text())
    train = [(e["step"], e["loss"]) for e in history if "loss" in e]
    evals = [(e["step"], e["eval_f1"]) for e in history if "eval_f1" in e]

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(*zip(*train))
    ax_loss.set(title="Training loss", xlabel="step", ylabel="loss")
    ax_f1.plot(*zip(*evals), marker="o")
    ax_f1.set(title="Validation entity F1", xlabel="step", ylabel="F1")
    fig.tight_layout()

    out = METRICS_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
