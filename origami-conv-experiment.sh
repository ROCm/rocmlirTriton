#!/usr/bin/env bash
#
# A/B the Origami quick-tune ranking on tier1 f16 convolutions. Runs tune +
# benchmark in a develop build, then in this (origami) build with the ranking
# uncropped and at ROCMLIR_ORIGAMI_TOP_N of 5, 10, 20 and 30, logging the wall
# time of each stage so the tuning-time saving can be weighed against whatever
# TFlops the crop gives up.
#
# Run it from the develop build directory:
#
#   cd /path/to/rocmlirDevelop/build
#   bash /path/to/rocmlirTriton/origami-conv-experiment.sh
#
# Every experiment writes its own tuning db, results file and logs, keyed on the
# experiment name, so nothing is overwritten and re-running the script resumes
# each experiment where it left off rather than re-timing it from scratch. To
# re-time one, delete its conv-tuning-<name>.tsv{,.state} first.
#
# See ROCMLIR_ORIGAMI_TOP_N in
# mlir/include/mlir/Dialect/Rock/Tuning/OrigamiRanker.h.

set -u

# Crop sizes to sweep on the origami side, against a 60-entry gfx1200 conv f16
# quick-tune list (gfx1201 falls back to it).
TOP_NS=(5 10 20 30)

# Both sides read the config list from this tree, so develop does not need the
# new tier1-conv-f16-configs file and every run is guaranteed the same input.
ORIGAMI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGAMI_BUILD="$ORIGAMI_ROOT/build"
CONFIGS="$ORIGAMI_ROOT/mlir/utils/performance/configs/tier1-conv-f16-configs"
DEVELOP_BUILD="$PWD"
LOGS="$PWD/origami-conv-experiment-logs"

# %3lR is elapsed wall time; the tuningRunner/perfRunner logs land in the same
# file, so each log ends with both the script's own timing line and this one.
export TIMEFORMAT='TOTAL  real %3lR  user %3lU  sys %3lS'

for path in "$DEVELOP_BUILD/bin/tuningRunner.py" "$ORIGAMI_BUILD/bin/tuningRunner.py" \
    "$CONFIGS"; do
  if [[ ! -e "$path" ]]; then
    echo "error: missing $path" >&2
    exit 1
  fi
done

mkdir -p "$LOGS"
echo "develop build : $DEVELOP_BUILD"
echo "origami build : $ORIGAMI_BUILD"
echo "configs       : $CONFIGS ($(wc -l < "$CONFIGS") configs)"
echo "logs          : $LOGS"
echo "experiments   : develop origami ${TOP_NS[*]/#/origami-top}"
echo

# run_experiment <name> <build dir> [top-n]
#
# The crop only affects tuning, so ROCMLIR_ORIGAMI_TOP_N is scoped to the
# tuningRunner invocation; perfRunner just replays the resulting tuning db.
run_experiment() {
  local name=$1 build=$2 top_n=${3:-}
  local tuning="conv-tuning-$name.tsv" results="conv-results-$name.tsv"
  local -a tuning_env=(env)

  if [[ -n $top_n ]]; then
    tuning_env+=("ROCMLIR_ORIGAMI_TOP_N=$top_n")
  fi

  cd "$build" || exit 1

  echo "=== $name: tuning starting $(date -Is)${top_n:+ (ROCMLIR_ORIGAMI_TOP_N=$top_n)} ==="
  { time "${tuning_env[@]}" ./bin/tuningRunner.py --op conv --tuning-space quick \
      --configs_file="$CONFIGS" -o "$tuning"; } 2>&1 |
    tee "$LOGS/$name-tuning.log"

  echo "=== $name: benchmarking starting $(date -Is) ==="
  { time python3 ./bin/perfRunner.py --op=conv --batch_mlir \
      --configs_file="$CONFIGS" --tuning_db="$tuning" \
      -o "$results"; } 2>&1 |
    tee "$LOGS/$name-perf.log"

  cp -f "$tuning" "$LOGS/$tuning" 2>/dev/null
  cp -f "$results" "$LOGS/$results" 2>/dev/null
  echo
}

run_experiment develop "$DEVELOP_BUILD"
run_experiment origami "$ORIGAMI_BUILD"
for top_n in "${TOP_NS[@]}"; do
  run_experiment "origami-top$top_n" "$ORIGAMI_BUILD" "$top_n"
done

echo "=== wall times ==="
grep -H -e '^TOTAL' -e 'completed successfully in' "$LOGS"/*.log
