"""Extract disease mentions from text with the fine-tuned model.

Research prototype only -- predictions are not clinical facts.

Usage:
    python src/predict.py "Patients with breast cancer often carry BRCA1 mutations."
    echo "some text" | python src/predict.py
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import pipeline

MODEL_DIR = Path(__file__).resolve().parent.parent / "outputs" / "final_model"


def load_predictor(model_dir: Path = MODEL_DIR):
    # "first" matches training: each word takes the label of its first subword.
    return pipeline("token-classification", model=str(model_dir), aggregation_strategy="first")


def predict(text: str, predictor) -> dict:
    entities = [
        {
            "text": text[e["start"]:e["end"]],
            "label": e["entity_group"],
            "start": int(e["start"]),
            "end": int(e["end"]),
            "score": round(float(e["score"]), 4),
        }
        for e in predictor(text)
    ]
    return {"entities": entities, "note": "Model predictions, not verified clinical findings."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="text to analyse (reads stdin if omitted)")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    print(json.dumps(predict(text, load_predictor(args.model_dir)), indent=2))


if __name__ == "__main__":
    main()
