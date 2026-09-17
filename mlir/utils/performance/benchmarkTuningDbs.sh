#!/usr/bin/env bash
#
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Benchmark every tuning database a tuningRunner.py run left behind.
#
# tuningRunner.py writes one `conv-tuning-<name>.tsv` per strategy, holding the
# perf config it settled on for each problem. That file says what was chosen,
# not what it is worth on this machine today: the number beside it was measured
# mid-search, against whatever else the search was doing at the time. Replaying
# the whole set through perfRunner.py in one pass is what makes two strategies
# comparable, and it lands in `result-conv-tuning-<name>.tsv`.
#
# Run it from wherever the tuning databases are -- the build directory, if that
# is where tuningRunner.py wrote them.
#
#   ./benchmarkTuningDbs.sh
#   ./benchmarkTuningDbs.sh -c ../mlir/utils/performance/configs/tier1-conv-configs
#   ./benchmarkTuningDbs.sh -- --flush-last-level-cache
#
set -euo pipefail

runner=./bin/perfRunner.py
configs=../mlir/utils/performance/configs/tier1-conv-exhaustive-wins-configs
directory=.
operation=conv
force=0

usage() {
    cat <<'EOF'
Usage: benchmarkTuningDbs.sh [options] [-- extra perfRunner.py arguments]

  -d DIR      Where the conv-tuning-*.tsv files are (default: .)
  -r PATH     perfRunner.py to run (default: ./bin/perfRunner.py)
  -c FILE     Config list to benchmark
              (default: ../mlir/utils/performance/configs/tier1-conv-configs)
  -o OP       Operation to pass as --op (default: conv)
  -f          Re-benchmark databases whose result file already exists
  -h          This message

Everything after `--` is handed to perfRunner.py untouched.
EOF
}

while getopts ':d:r:c:o:fh' option; do
    case "$option" in
        d) directory=$OPTARG ;;
        r) runner=$OPTARG ;;
        c) configs=$OPTARG ;;
        o) operation=$OPTARG ;;
        f) force=1 ;;
        h) usage; exit 0 ;;
        :) echo "error: -$OPTARG needs a value" >&2; usage >&2; exit 2 ;;
        \?) echo "error: unknown option -$OPTARG" >&2; usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

[[ -x $runner || -f $runner ]] || { echo "error: no perfRunner.py at $runner" >&2; exit 2; }
[[ -f $configs ]] || { echo "error: no config list at $configs" >&2; exit 2; }
[[ -d $directory ]] || { echo "error: no directory $directory" >&2; exit 2; }

# `conv-tuning-*.tsv` and nothing else. A run leaves `.debug`, `.state` and a
# `.search` directory beside each database, and the pattern ends at `.tsv`, so
# none of those match -- `conv-tuning-llm.tsv.debug` is a different name, not a
# `.tsv` with a suffix. The `-f` below says the same thing a second way, since
# `.search` is a directory and a glob cannot tell that on its own. An earlier
# pass of this script wrote `result-conv-tuning-*.tsv`, which the prefix keeps
# out.
shopt -s nullglob
databases=()
for candidate in "$directory"/conv-tuning-*.tsv; do
    [[ -f $candidate ]] && databases+=("$candidate")
done
shopt -u nullglob

if [[ ${#databases[@]} -eq 0 ]]; then
    echo "no conv-tuning-*.tsv in $directory" >&2
    exit 1
fi

benchmarked=()
skipped=()
failed=()

for database in "${databases[@]}"; do
    name=$(basename "$database")
    result="$directory/result-$name"

    # A tuning run that died early leaves a file with a header and no rows, or
    # no file content at all. Benchmarking it would produce an empty result
    # that reads like a strategy that lost everywhere.
    if [[ ! -s $database ]] || ! grep -qv '^#' "$database"; then
        echo "== $name: no tuned configs in it, skipping"
        skipped+=("$name")
        continue
    fi

    if [[ -e $result && $force -eq 0 ]]; then
        echo "== $name: $(basename "$result") is already there, skipping (-f to redo)"
        skipped+=("$name")
        continue
    fi

    rows=$(grep -cv '^#' "$database" || true)
    echo "== $name: benchmarking $rows tuned configs -> $(basename "$result")"

    # `-b` benchmarks the MLIR kernels alone, which is what the tuning database
    # holds configs for; the external reference is a separate question.
    if "$runner" -b \
        --op "$operation" \
        --configs-file "$configs" \
        --tuning-db "$database" \
        -o "$result" \
        "$@"; then
        benchmarked+=("$name")
    else
        status=$?
        echo "== $name: perfRunner.py exited $status" >&2
        failed+=("$name")
    fi
done

echo
echo "benchmarked ${#benchmarked[@]}, skipped ${#skipped[@]}, failed ${#failed[@]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    printf 'failed: %s\n' "${failed[*]}" >&2
    exit 1
fi
