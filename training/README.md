# MediKo NER training

Fine-tunes three BERT-base models for disease NER on the NCBI Disease corpus,
with the same data, hyperparameters and seed, and compares them:

| Key | Model | Pretraining |
|---|---|---|
| `bert` | `google-bert/bert-base-cased` | general English, cased |
| `biobert` | `dmis-lab/biobert-base-cased-v1.2` | BERT further pretrained on PubMed, cased |
| `biomedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | PubMedBERT, PubMed from scratch, uncased |

This is the baseline for the spec in `../medical_report_intelligence_spec.md`.
It is separate from the inference app and has its own image because it needs
PyTorch. The models are listed in `src/models.py`.

## Run

Needs Docker only (Docker Desktop on Windows). The launcher checks whether Docker
has the NVIDIA runtime and trains on the GPU if so, otherwise on the CPU.

```powershell
cd training
.\run.ps1            # Windows (PowerShell)
```

```bash
./run.sh             # Linux / macOS
```

The first run builds the image (several minutes, the CUDA PyTorch wheel is large),
downloads the dataset and the three models (cached in `./data` and a Docker
volume), trains each model in turn, evaluates each on validation and test, and
prints a comparison table at the end. Everything is written to `./outputs/`.
The default is 3 epochs. Each model takes about as long as the single-model
run did: on a GTX 1650 Ti, 4 epochs took about 10 minutes; on a 6-core CPU,
3 epochs took about an hour. Three models take about three times that.

To train only some models, set `MODELS` (space-separated keys):

```powershell
$env:MODELS = "bert biobert"; .\run.ps1
```

```bash
MODELS="bert biobert" ./run.sh
```

If one model fails, the others still train, the comparison shows the ones that
finished, and the run exits with an error that names the failed model.

A short demo run (a few minutes on GPU):

```powershell
.\run.ps1 --epochs 3 --limit 800
```

Arguments are passed to `src/train.py` for every model. The container is removed when it
finishes, so read the results from `./outputs/` on the host.

On a GPU with less than 4 GB, add `--batch-size 4 --grad-accum 4`. That keeps
the effective batch size at 16.

Without the launcher:

```bash
docker compose run --rm train                                      # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm train   # GPU
```

## What you get in `outputs/`

Comparison of all models, in `outputs/`:

| File | Contents |
|---|---|
| `comparison.txt` | the comparison table printed at the end of the run |
| `comparison.csv`, `comparison.json` | the same numbers, per model |
| `comparison.png` | validation F1 per epoch for each model, and test P/R/F1 bars |

Per model, in `outputs/<model>/` (`bert`, `biobert`, `biomedbert`):

| File | Contents |
|---|---|
| `train.log` | full console log of that model's run |
| `metrics/log_history.json` | loss every 10 steps, plus validation metrics each epoch |
| `metrics/training_curves.png` | training loss and validation F1 plot |
| `metrics/validation.json`, `test.json` | final metrics (best checkpoint) |
| `metrics/test_classification_report.txt` | per-label precision, recall, F1 |
| `metrics/train_config.json` | model, hyperparameters and seed |
| `final_model/` | best model and tokenizer |

Each model also keeps up to two checkpoints in `checkpoints/` (about 1.3 GB
each, with optimizer state). Delete them after the run if disk space is short;
`final_model/` is all that `predict.py` needs.

## Method

- Data: NCBI Disease, CoNLL format, labels `O`, `B-Disease`, `I-Disease`. The
  train, devel and test splits come from the corpus and are fixed.
- Labels are aligned to the first subword of each word. Other subwords and
  special tokens are ignored in the loss.
- The best epoch is picked by validation entity F1. The test set is evaluated
  once, after selection. Metrics are entity-level (`seqeval`).
- All models use the same hyperparameters and seed. The comparison warns if the
  runs it reads used different settings (for example a `--limit` run mixed with
  a full one).
- BioBERT's repository has no tokenizer config, so its tokenizer is loaded with
  `do_lower_case=False`. Without that it would lowercase text for a cased
  vocabulary.
- Each model is one run with one seed. Small differences in test F1 can be
  run-to-run noise.
- Reference run, BiomedBERT only (seed 42, 3 epochs): test P 0.848, R 0.890,
  F1 0.868.

## Limits

NCBI Disease is biomedical literature, not clinical reports, and has one entity
type. It does not yet test noisy or OCR text.

## Inference and tests

```bash
python src/predict.py "Patients with breast cancer often carry BRCA1 mutations."   # BiomedBERT
python src/predict.py --model bert "..."
python src/compare_models.py          # rebuild the comparison from outputs/
python -m unittest discover -s tests
```
