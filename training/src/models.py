"""Base models compared on NCBI Disease. The keys are what MODELS and --model accept.

All three are BERT-base (12 layers, 768 hidden) and differ only in pretraining:
- bert:       general English (Wikipedia + books), cased.
- biobert:    BERT-base weights and vocabulary, further pretrained on PubMed, cased.
- biomedbert: PubMedBERT, pretrained from scratch on PubMed with its own vocabulary, uncased.
"""

MODELS = {
    "bert": {
        "id": "google-bert/bert-base-cased",
        "name": "BERT",
        "tokenizer_kwargs": {},
    },
    "biobert": {
        "id": "dmis-lab/biobert-base-cased-v1.2",
        "name": "BioBERT",
        # The repo has no tokenizer_config.json, so the tokenizer would default to
        # lowercasing and feed lowercased text to a cased vocabulary.
        "tokenizer_kwargs": {"do_lower_case": False},
    },
    "biomedbert": {
        "id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "name": "BiomedBERT (PubMedBERT)",
        "tokenizer_kwargs": {},
    },
}
DEFAULT_MODEL = "biomedbert"


def load_tokenizer(key: str):
    from transformers import AutoTokenizer

    spec = MODELS[key]
    return AutoTokenizer.from_pretrained(spec["id"], **spec["tokenizer_kwargs"])
