#!/usr/bin/env bash

# Benchmark every unsuccessful convolution test vector except the first,
# already-investigated c512/H256/W256/k512 problem. Run this inside the
# mh-triton container; the paths below are the container-visible paths.

set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/mhusic/rocmlirTriton}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT_DIR}/convolution_tuning_benchmark_first-10-08/results.tsv}"
STATE_FILE="${STATE_FILE:-${RESULTS_FILE}.state}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/home/mhusic/convolution_tuning_MEDUSA_first-10-08/artifacts}"
SKIP_FILE="${SKIP_FILE:-${ROOT_DIR}/gfx1170-skip-conv.txt}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/convolution-error-logs-18-08}"
CONTEXT="${CONTEXT:-gfx1170/4/1/exhaustive}"

SKIP_TV='conv -F 1 -f GNC01 -I NGC01 -O NGC01 -n 1 -c 512 -H 256 -W 256 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1'

if [[ ! -f "${BUILD_DIR}/bin/tuningRunner.py" ]]; then
  echo "error: tuningRunner.py not found under ${BUILD_DIR}/bin" >&2
  exit 1
fi
if [[ ! -f "${STATE_FILE}" ]]; then
  echo "error: state file not found: ${STATE_FILE}" >&2
  exit 1
fi
if [[ ! -d "${ARTIFACTS_DIR}" ]]; then
  echo "error: artifacts directory not found: ${ARTIFACTS_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SKIP_FILE}" ]]; then
  echo "error: skip list not found: ${SKIP_FILE}" >&2
  exit 1
fi
if [[ ! -x "${ROCM_PATH:-/opt/rocm}/bin/rocm-smi" ]]; then
  echo "error: no rocm-smi under ROCM_PATH=${ROCM_PATH:-/opt/rocm}" >&2
  echo "       re-activate the rocm-sdk venv, or set ROCM_PATH by hand" >&2
  exit 1
fi
if command -v pgrep >/dev/null &&
   pgrep -f '[r]ocmlir-tuning-driver|[b]in/tuningRunner.py.*--benchmark-artifacts' \
     >/dev/null; then
  echo "error: another artifact benchmark appears to be running" >&2
  echo "Stop it before starting this sequential run." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

if ! config_output="$(
  python3 - "${STATE_FILE}" "${CONTEXT}" "${SKIP_TV}" <<'PY'
import json
import sys

state_file, context, skip_tv = sys.argv[1:]
selected_states = {
    "failed",
    "timed_out",
    "gpu_timed_out",
    "crashed",
    "running",
    "interrupted",
}

with open(state_file, encoding="utf-8") as handle:
    state = json.load(handle)

contexts = state.get("contexts", {})
if context not in contexts:
    available = ", ".join(contexts) or "<none>"
    raise SystemExit(
        f"error: context {context!r} is absent from {state_file}; "
        f"available contexts: {available}"
    )

for test_vector, status in contexts[context].items():
    if test_vector != skip_tv and status in selected_states:
        print(test_vector)
PY
)"; then
  exit 1
fi

CONFIGS=()
if [[ -n "${config_output}" ]]; then
  mapfile -t CONFIGS <<<"${config_output}"
fi

if (( ${#CONFIGS[@]} == 0 )); then
  echo "No unsuccessful convolution test vectors remain after exclusions."
  exit 0
fi

echo "Skipping the already-investigated first problem:"
echo "  ${SKIP_TV}"
echo
echo "Will benchmark ${#CONFIGS[@]} remaining problem(s) sequentially."
echo "A nonzero result is logged and does not stop the next problem."

stop_requested=0
trap 'stop_requested=1' INT TERM

completed=0
unsuccessful=0
for index in "${!CONFIGS[@]}"; do
  tv="${CONFIGS[index]}"
  sequence=$((index + 1))

  slug="$(
    python3 - "${tv}" <<'PY'
import hashlib
import shlex
import sys

test_vector = sys.argv[1]
tokens = shlex.split(test_vector)
params = {
    tokens[index]: tokens[index + 1]
    for index in range(1, len(tokens) - 1, 2)
}
digest = hashlib.sha256(test_vector.encode()).hexdigest()[:8]
print(
    f"{tokens[0]}-c{params.get('-c', 'x')}"
    f"-h{params.get('-H', 'x')}-w{params.get('-W', 'x')}"
    f"-k{params.get('-k', 'x')}-{digest}"
)
PY
  )"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log_file="${LOG_DIR}/$(printf '%02d' "${sequence}")-${slug}-${timestamp}.log"

  echo
  echo "[$(printf '%02d' "${sequence}")/${#CONFIGS[@]}] ${tv}"
  echo "Log: ${log_file}"

  python3 "${BUILD_DIR}/bin/tuningRunner.py" \
    --allow-commit-mismatch \
    --op conv \
    --config "${tv}" \
    --tuning-space=exhaustive \
    --benchmark-artifacts="${ARTIFACTS_DIR}" \
    --skip-perf-configs "${SKIP_FILE}" \
    --debug-quick-tune-data \
    --gpu-run-timeout 60 \
    -o "${RESULTS_FILE}" \
    --retry failed timed_out crashed gpu_timed_out \
    2>&1 | tee "${log_file}"
  runner_status=${PIPESTATUS[0]}

  if (( stop_requested || runner_status == 130 )); then
    echo "Interrupted; stopping before the next problem." >&2
    exit 130
  fi

  completed=$((completed + 1))
  if (( runner_status != 0 )); then
    unsuccessful=$((unsuccessful + 1))
    echo "Problem exited with status ${runner_status}; continuing."
  fi
done

echo
echo "Finished ${completed} problem(s); ${unsuccessful} exited unsuccessfully."
echo "Logs: ${LOG_DIR}"
