"""Download the NCBI Disease corpus (CoNLL TSV) and the BiomedBERT model.

Usage:
    python src/download_data.py               # dataset + model
    python src/download_data.py --skip-model  # dataset only
    python src/download_data.py --skip-data   # model only
"""

import argparse
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ncbi_disease"

BASE_URL = "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll"
SPLITS = ["train", "devel", "test"]

MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
LABELS = ["O", "B-Disease", "I-Disease"]


def download_file(url: str, dest: Path, retries: int = 3) -> None:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                content = resp.read()
            if not content.strip():
                raise ValueError(f"Downloaded file is empty: {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(content)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, ValueError, TimeoutError) as err:
            last_err = err
            print(f"  attempt {attempt}/{retries} failed: {err}")
    raise RuntimeError(f"Could not download {url}: {last_err}")


def validate_tsv(path: Path) -> tuple[int, int, Counter]:
    """Check the token<TAB>tag format. Returns (sentences, tokens, tag counts)."""
    sentences, tokens, in_sentence = 0, 0, False
    tags: Counter = Counter()
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                if in_sentence:
                    sentences += 1
                in_sentence = False
                continue
            parts = line.split("\t")
            if len(parts) != 2 or not parts[0]:
                raise ValueError(f"{path.name}:{lineno}: malformed row {line!r}")
            if parts[1] not in LABELS:
                raise ValueError(f"{path.name}:{lineno}: unknown tag {parts[1]!r}")
            tags[parts[1]] += 1
            tokens += 1
            in_sentence = True
    if in_sentence:  # file without a trailing blank line
        sentences += 1
    if sentences == 0:
        raise ValueError(f"{path.name}: no sentences found")
    return sentences, tokens, tags


def download_dataset(force: bool) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset -> {DATA_DIR}")
    for split in SPLITS:
        dest = DATA_DIR / f"{split}.tsv"
        if dest.exists() and not force:
            print(f"  {dest.name}: already present (use --force to re-download)")
        else:
            print(f"  downloading {split}.tsv ...")
            download_file(f"{BASE_URL}/{split}.tsv", dest)
        sents, toks, tags = validate_tsv(dest)
        tag_str = ", ".join(f"{t}={tags[t]}" for t in LABELS)
        print(f"  {dest.name}: {sents} sentences, {toks} tokens ({tag_str})")


def download_model() -> None:
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        sys.exit("transformers is not installed. Run: pip install -r requirements.txt")

    print(f"Model -> {MODEL_ID} (cached in the Hugging Face cache, ~440 MB)")
    id2label = dict(enumerate(LABELS))
    label2id = {label: i for i, label in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # The token-classification head is newly initialised; the warning about
    # "newly initialized weights" is expected -- fine-tuning trains that head.
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )

    # Quick check that tokenizer + model work together.
    words = ["Mutations", "in", "BRCA1", "cause", "breast", "cancer", "."]
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt")
    logits = model(**enc).logits
    print(f"  tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"  subword tokens: {tokenizer.convert_ids_to_tokens(enc['input_ids'][0])}")
    print(f"  forward pass OK, logits shape: {tuple(logits.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-data", action="store_true", help="do not download the dataset")
    parser.add_argument("--skip-model", action="store_true", help="do not download the model")
    parser.add_argument("--force", action="store_true", help="re-download dataset files")
    args = parser.parse_args()

    if not args.skip_data:
        download_dataset(args.force)
    if not args.skip_model:
        download_model()
    print("Done.")


if __name__ == "__main__":
    main()
