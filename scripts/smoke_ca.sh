#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs/smoke-ca}"
python ca_workflow.py generate --root "$ROOT" --items 20 --bids 80 --train 4 --valid 2 --test 2 --seed 1700
python ca_workflow.py collect --root "$ROOT" --workers 2 --time-limit 5 --max-solutions 10
python ca_workflow.py train --root "$ROOT" --epochs 2 --batch-size 2 --device cuda
for SOLVER in gurobi scip; do
  for MODE in baseline search fixing; do
    DELTA=0
    if [ "$MODE" = search ]; then DELTA=2; fi
    python ca_workflow.py evaluate --root "$ROOT" --name "$SOLVER-$MODE" --solver "$SOLVER" --mode "$MODE" --time-limit 5 --k0 20 --k1 0 --delta "$DELTA"
  done
done
python ca_workflow.py summarize --reference "$ROOT/evaluation/gurobi-baseline/results.json" "$ROOT"/evaluation/*/results.json --output "$ROOT/summary.json"
