# MediKo NER training

Fine-tunes `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (the
PubMedBERT model) for disease NER on the NCBI Disease corpus. This is the
baseline for the spec in `../medical_report_intelligence_spec.md`. It is
separate from the inference app and has its own image because it needs PyTorch.

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
downloads the dataset and model (cached in `./data` and a Docker volume), trains,
evaluates on validation and test, and writes everything to `./outputs/`. The
default is 3 epochs. On a GTX 1650 Ti, 4 epochs took about 10 minutes. On a
6-core CPU, 3 epochs took about an hour.

A short demo run (a few minutes on GPU):

```powershell
.\run.ps1 --epochs 3 --limit 800
```

Arguments are passed to `src/train.py`. The container is removed when it
finishes, so read the results from `./outputs/` on the host.

On a GPU with less than 4 GB, add `--batch-size 4 --grad-accum 4`. That keeps
the effective batch size at 16.

Without the launcher:

```bash
docker compose run --rm train                                      # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm train   # GPU
```

## What you get in `outputs/`

| File | Contents |
|---|---|
| `train.log` | full console log of the run |
| `metrics/log_history.json` | loss every 10 steps, plus validation metrics each epoch |
| `metrics/training_curves.png` | training loss and validation F1 plot |
| `metrics/validation.json`, `test.json` | final metrics (best checkpoint) |
| `metrics/test_classification_report.txt` | per-label precision, recall, F1 |
| `metrics/train_config.json` | hyperparameters and seed |
| `final_model/` | best model and tokenizer |

## Method

- Data: NCBI Disease, CoNLL format, labels `O`, `B-Disease`, `I-Disease`. The
  train, devel and test splits come from the corpus and are fixed.
- Labels are aligned to the first subword of each word. Other subwords and
  special tokens are ignored in the loss.
- The best epoch is picked by validation entity F1. The test set is evaluated
  once, after selection. Metrics are entity-level (`seqeval`).
- Reference run (seed 42, 3 epochs): test P 0.848, R 0.890, F1 0.868.

## Limits

NCBI Disease is biomedical literature, not clinical reports, and has one entity
type. It does not yet test noisy or OCR text.

## Inference and tests

```bash
python src/predict.py "Patients with breast cancer often carry BRCA1 mutations."
python -m unittest discover -s tests
```
