#!/usr/bin/env bash
# N full runs then one plot across them. extra args go to bench.py, e.g. ./run.sh --mode reduce-overhead
set -euo pipefail
cd "$(dirname "$0")"
N=${N_RUNS:-3}
STAMP=$(date +%Y%m%d-%H%M%S)
dirs=()
for i in $(seq 1 "$N"); do
  python bench.py --run-id "$STAMP-r$i" "$@"
  dirs+=("results/$STAMP-r$i")
done
python plot.py "${dirs[@]}" --out "plots/$STAMP"
cp "results/$STAMP-r1/snapshot.json" "plots/$STAMP/"
echo "done: plots/$STAMP"
