"""Parse the NCBI Disease CoNLL TSV files and align word labels to subword tokens.

Alignment strategy: the first subword of each word gets the word's label;
continuation subwords and special tokens ([CLS], [SEP], padding) get -100 so
the loss and metrics ignore them.
"""

from pathlib import Path

LABELS = ["O", "B-Disease", "I-Disease"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
IGNORE_INDEX = -100

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ncbi_disease"
SPLIT_FILES = {"train": "train.tsv", "validation": "devel.tsv", "test": "test.tsv"}


def read_conll(path: Path) -> list[dict]:
    """Read token<TAB>tag lines into [{"tokens": [...], "ner_tags": [ids]}]."""
    examples, tokens, tags = [], [], []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                if tokens:
                    examples.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) != 2 or not parts[0] or parts[1] not in LABEL2ID:
                raise ValueError(f"{path}:{lineno}: malformed row {line!r}")
            tokens.append(parts[0])
            tags.append(LABEL2ID[parts[1]])
    if tokens:  # final example without a trailing blank line
        examples.append({"tokens": tokens, "ner_tags": tags})
    if not examples:
        raise ValueError(f"{path}: no examples found")
    return examples


def align_labels(word_ids: list, word_labels: list[int]) -> list[int]:
    labels, previous = [], None
    for word_id in word_ids:
        if word_id is None or word_id == previous:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(word_labels[word_id])
        previous = word_id
    return labels


def tokenize_and_align(batch: dict, tokenizer, max_length: int) -> dict:
    enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
    enc["labels"] = [
        align_labels(enc.word_ids(i), word_labels) for i, word_labels in enumerate(batch["ner_tags"])
    ]
    return enc


def load_splits(data_dir: Path = DATA_DIR):
    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {split: Dataset.from_list(read_conll(Path(data_dir) / name)) for split, name in SPLIT_FILES.items()}
    )
