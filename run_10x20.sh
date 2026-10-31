#!/usr/bin/env bash
# Reproduce every result of the study on the standard 10x20 board:
#   - the reference agents (Random, single-feature greedy, Heuristic),
#   - the 2x2 grid: CE linear / CE MLP (policy search) and linear TD / DQN
#     (value learning), all trained on the number of lines cleared,
#   - the figures and the summary table.
# Results go to results_10x20/. Runs are deterministic given the seed, so the
# number of parallel jobs and CE workers does not change them.
#
# Usage: bash run_10x20.sh [PYTHON]
#   SEEDS       seeds of every model (default: 0 1 2)
#   CE_WORKERS  worker processes per CE run (default: 8)
set -uo pipefail
cd "$(dirname "$0")"
PY=${1:-python}
OUT=results_10x20
SEEDS=${SEEDS:-0 1 2}
CE_WORKERS=${CE_WORKERS:-8}
COMMON=(--height 20 --out "$OUT")
mkdir -p "$OUT/logs"

"$PY" -m t2x2.baselines "${COMMON[@]}" > "$OUT/logs/baselines.log" 2>&1 &

# Value learning: single-threaded runs in parallel.
for seed in $SEEDS; do
  "$PY" -m t2x2.train_td --network linear --seed "$seed" "${COMMON[@]}" > "$OUT/logs/linear_td_$seed.log" 2>&1 &
  "$PY" -m t2x2.train_td --network mlp --seed "$seed" "${COMMON[@]}" > "$OUT/logs/dqn_$seed.log" 2>&1 &
done
wait

# Policy search: each run already uses a worker pool, so run them one by one.
for seed in $SEEDS; do
  for policy in linear mlp; do
    "$PY" -m t2x2.train_ce --policy "$policy" --seed "$seed" --workers "$CE_WORKERS" \
      "${COMMON[@]}" > "$OUT/logs/ce_${policy}_$seed.log" 2>&1
  done
done

"$PY" -m t2x2.plot --results "$OUT" --board "10×20"
echo "done $(date)"
