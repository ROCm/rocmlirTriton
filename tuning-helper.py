#!/usr/bin/env python3
"""Query an artifact bundle against a results TSV.

Two subcommands, both used by run-gemm-attention-overnight.sh:

  pending <bundle> <results>
      Print the test vector of every problem in the bundle that has no winner
      in the results TSV, one per line.

  known-good-skip <bundle> <results> <out>
      Write a --skip-perf-configs file covering every config compiled for those
      pending problems that has never won on any already-tuned problem. What is
      left is a candidate set known to run cleanly somewhere, which is how a
      problem that keeps hanging the GPU can still produce a winner.

Bundle manifests are zstd with an 8-byte little-endian length prefix.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path


def read_manifest(path):
    payload = Path(path).read_bytes()[8:]
    done = subprocess.run(["zstd", "-d", "-c"], input=payload, capture_output=True, check=True)
    return json.loads(done.stdout)


def read_results(results):
    """Winner rows, minus the header lines that repeat when runs are appended."""
    if not Path(results).exists():
        return []
    with open(results, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [r for r in rows if r.get("perfConfig") and r["perfConfig"] != "perfConfig"]


def find_pending(bundle, results):
    index = json.loads((Path(bundle) / "index.json").read_text())
    tuned = {r["testVector"] for r in read_results(results)}
    return [(h, v["testVector"]) for h, v in index["problems"].items()
            if v["testVector"] not in tuned]


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    command, bundle, results = sys.argv[1:4]
    pending = find_pending(bundle, results)

    if command == "pending":
        for _, test_vector in pending:
            print(test_vector)
        return

    if command != "known-good-skip":
        sys.exit(f"unknown command: {command}")

    out = sys.argv[4]
    winners = {r["perfConfig"] for r in read_results(results)}

    space = set()
    for problem_hash, _ in pending:
        manifest = read_manifest(Path(bundle) / "problems" / problem_hash / "manifest.json.z")
        space.update(c["perfConfig"] for c in manifest["configs"] if c["status"] == "success")

    skip = sorted(space - winners)
    with open(out, "w") as handle:
        handle.write(f"# known-good skip list for {bundle}\n")
        handle.write(f"# {len(pending)} pending problem(s), {len(space)} compiled configs\n")
        handle.write(f"# skipping {len(skip)} that never won on any tuned problem, "
                     f"leaving {len(space) - len(skip)} candidates\n")
        for config in skip:
            handle.write(config + "\n")

    print(f"{out}: {len(skip)} skipped, {len(space) - len(skip)} candidates "
          f"across {len(pending)} pending problem(s)")


if __name__ == "__main__":
    main()
