#!/bin/sh
# download dataset + models (skipped when already cached) -> train each model -> plot curves -> compare.
# MODELS picks the models (default: all three, see src/models.py), e.g. MODELS="bert biobert".
# Extra arguments go to src/train.py for every model, e.g. --epochs 1 --limit 200
set -eu
MODELS="${MODELS:-bert biobert biomedbert}"
mkdir -p /app/outputs
# shellcheck disable=SC2086
python src/download_data.py --models $MODELS

failed=""
for model in $MODELS; do
  out="/app/outputs/$model"
  # Old metrics must not show up in the comparison if this run fails.
  rm -rf "$out/metrics"
  mkdir -p "$out"
  echo ">> Training $model"
  # tee would hide train.py's exit status, so record it. One failed model does not stop the others.
  { rc=0; python src/train.py --model "$model" --output-dir "$out" "$@" 2>&1 || rc=$?; echo "$rc" > "$out/.exit"; } \
    | tee "$out/train.log"
  if [ "$(cat "$out/.exit")" -eq 0 ]; then
    python src/plot_training.py "$out/metrics"
  else
    failed="$failed $model"
  fi
  rm -f "$out/.exit"
done

# shellcheck disable=SC2086
python src/compare_models.py --models $MODELS 2>&1 | tee /app/outputs/comparison.txt
if [ -n "$failed" ]; then
  echo "Training failed for:$failed (see outputs/<model>/train.log)" >&2
  exit 1
fi
