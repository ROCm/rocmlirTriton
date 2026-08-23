#!/usr/bin/env bash
# Finish the GEMM and attention problems that have no winner yet on gfx1170.
#
# Replays the existing MEDUSA bundles, so nothing is compiled or rebuilt. Do NOT
# run cmake.sh first: it wipes build/, and the current build is the only one that
# has --skip-perf-configs, which pass 2 below depends on.
#
# Each problem runs in its own tuningRunner process, so a kernel that wedges the
# GPU costs one problem instead of the whole queue.
#
# Pass 1 benchmarks the full compiled space, which gives the true exhaustive
# winner. Pass 2 retries whatever pass 1 lost, restricted to configs that have
# already won on some other problem, so a problem that keeps hanging still ends
# up with a winner. Pass 2 is a no-op when pass 1 finishes everything.
#
# Run inside the mh-triton container with no other tuning in flight.

set -uo pipefail

SRC=/home/mhusic/rocmlirTriton
BUILD="$SRC/build"
LOG_DIR="$SRC/gemm-attention-logs-23-08"
TIMEOUT=60

mkdir -p "$LOG_DIR"

if [[ ! -f "$BUILD/bin/tuningRunner.py" ]]; then
  echo "error: $BUILD/bin/tuningRunner.py not found; do not rebuild, this run needs the existing build" >&2
  exit 1
fi
if pgrep -f '[r]ocmlir-tuning-driver|[b]in/tuningRunner.py' >/dev/null; then
  echo "error: tuning is already running; stop it before starting this" >&2
  exit 1
fi
# A polluted ROCM_PATH makes every problem die instantly on a missing rocm-smi,
# which looks like a tuning failure and wastes the whole run.
if [[ ! -x "${ROCM_PATH:-/opt/rocm}/bin/rocm-smi" ]]; then
  echo "error: no rocm-smi under ROCM_PATH=${ROCM_PATH:-/opt/rocm}" >&2
  echo "       re-activate the rocm-sdk venv, or set ROCM_PATH by hand" >&2
  exit 1
fi

# op | artifact bundle | results TSV
WORK=(
  "gemm|/home/mhusic/GEMM_tuning_MEDUSA/artifacts|$SRC/GEMM_tuning_benchmark/results.tsv"
  "attention|/home/mhusic/attention_tuning_MEDUSA/artifacts|$SRC/attention_tuning_benchmark/results.tsv"
)

# Benchmark every problem that still lacks a winner, one process each.
# $1 pass label, $2 op, $3 bundle, $4 results, $5 skip file ("" for none)
run_pass() {
  local pass="$1" op="$2" bundle="$3" results="$4" skip="$5"
  local pending count=0 index=0

  pending="$(python3 "$SRC/tuning-helper.py" pending "$bundle" "$results")" || return 1
  [[ -z "$pending" ]] && { echo "[$pass/$op] nothing pending"; return 0; }
  count="$(printf '%s\n' "$pending" | wc -l)"
  echo "[$pass/$op] $count problem(s) pending"

  while IFS= read -r test_vector; do
    index=$((index + 1))
    local log="$LOG_DIR/${pass}-${op}-$(printf '%02d' "$index")-$(date -u +%H%M%SZ).log"
    echo
    echo "[$pass/$op $index/$count] $test_vector"
    echo "  log: $log"

    python3 "$BUILD/bin/tuningRunner.py" \
      --op "$op" \
      --config "$test_vector" \
      --tuning-space=exhaustive \
      --benchmark-artifacts="$bundle" \
      --allow-commit-mismatch \
      ${skip:+--skip-perf-configs "$skip"} \
      --debug-quick-tune-data \
      --gpu-run-timeout "$TIMEOUT" \
      --retry failed timed_out crashed gpu_timed_out \
      -o "$results" \
      > "$log" 2>&1
    local status=$?

    if (( status == 0 )); then
      grep -h "Benchmarked problem" "$log" | sed 's/^/  /'
    else
      echo "  FAILED with status $status (see log); continuing"
      grep -h "presumed hung\|ERROR" "$log" | head -3 | sed 's/^/  /'
    fi
  done <<< "$pending"
}

for entry in "${WORK[@]}"; do
  IFS='|' read -r op bundle results <<< "$entry"
  echo
  echo "############ $op: pass 1, full compiled space ############"
  run_pass pass1 "$op" "$bundle" "$results" ""
done

for entry in "${WORK[@]}"; do
  IFS='|' read -r op bundle results <<< "$entry"
  echo
  echo "############ $op: pass 2, known-good configs only ############"
  skip="$SRC/skip-known-good-$op.txt"
  if python3 "$SRC/tuning-helper.py" known-good-skip "$bundle" "$results" "$skip"; then
    run_pass pass2 "$op" "$bundle" "$results" "$skip"
  else
    echo "  could not build $skip; skipping pass 2 for $op"
  fi
done

echo
echo "############ summary ############"
for entry in "${WORK[@]}"; do
  IFS='|' read -r op bundle results <<< "$entry"
  left="$(python3 "$SRC/tuning-helper.py" pending "$bundle" "$results" | wc -l)"
  echo "$op: $left problem(s) still without a winner"
  echo "  winners: $results"
  echo "  per-config data for the quick-tune list: $results.debug"
done
