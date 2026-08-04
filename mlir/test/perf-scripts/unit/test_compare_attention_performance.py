# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: env PYTHONPATH=%S/../../../utils/performance %python %s

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import compareAttentionPerformance

comparison = compareAttentionPerformance


def sample(label="base",
           sha="base-sha",
           chip="gfx90a",
           config="-t f16 -g 1 -seq_len_q 4 -seq_len_k 4 -head_dim_qk 8 -head_dim_v 8",
           number=1,
           perf_config="v1",
           tflops="10"):
    return {
        "RunLabel": label,
        "SourceSha": sha,
        "Chip": chip,
        "Config": config,
        "RocmlirGenFlags": "",
        "Sample": str(number),
        "PerfConfig": perf_config,
        "TFlops": tflops,
    }


class CompareAttentionPerformanceTest(unittest.TestCase):

    def test_read_and_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            fields = list(sample())
            for path, row in ((first, sample(number=1)), (second, sample(number=2, tflops="14"))):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(row)
            rows = comparison.read_csv_rows([first, second])
        aggregated = comparison.aggregate_samples(rows, "base")
        result = next(iter(aggregated.values()))
        self.assertEqual(result["tflops"], 12)
        self.assertEqual(result["samples"], 2)

    def test_aggregate_rejects_bad_data(self):
        with self.assertRaisesRegex(ValueError, "RunLabel"):
            comparison.aggregate_samples([sample(label="candidate")], "base")
        with self.assertRaisesRegex(ValueError, "Duplicate sample"):
            comparison.aggregate_samples([sample(), sample()], "base")
        with self.assertRaisesRegex(ValueError, "exactly one source revision"):
            comparison.aggregate_samples([sample(), sample(sha="other", number=2)], "base")
        for changed in ("PerfConfig", "RocmlirGenFlags"):
            first = sample(number=1)
            second = sample(number=2)
            if changed == "PerfConfig":
                second["PerfConfig"] = "other"
            else:
                second["RocmlirGenFlags"] = "-flag"
            with self.assertRaisesRegex(ValueError, "Inconsistent"):
                comparison.aggregate_samples([first, second], "base")
        for value in ("nan", "0", "-1"):
            with self.assertRaisesRegex(ValueError, "Invalid TFlops"):
                comparison.aggregate_samples([sample(tflops=value)], "base")

    def test_compare_and_config_columns(self):
        base = [sample(number=1, tflops="10"), sample(number=2, tflops="14")]
        candidate = [
            sample(label="candidate", sha="candidate-sha", number=1, tflops="12"),
            sample(label="candidate", sha="candidate-sha", number=2, tflops="16")
        ]
        rows = comparison.compare_results(base, candidate, 2)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["% Diff"], 100 * (14 - 12) / 12)
        self.assertEqual(rows[0]["DataType"], "f16")
        self.assertEqual(rows[0]["SplitKV"], "1")
        self.assertEqual(rows[0]["TransQ"], "false")

        with self.assertRaisesRegex(ValueError, "keys differ"):
            comparison.compare_results(
                base, candidate +
                [sample(label="candidate", sha="candidate-sha", chip="gfx950", number=1)], 2)
        candidate[1]["Sample"] = "3"
        with self.assertRaisesRegex(ValueError, "Sample IDs differ"):
            comparison.compare_results(base, candidate, 2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            comparison.compare_results([base[0]], [candidate[0]], 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            comparison.compare_results(base, candidate, 0)
        candidate[1]["Sample"] = "2"
        candidate[0]["RocmlirGenFlags"] = "-current_seq_len=3"
        candidate[1]["RocmlirGenFlags"] = "-current_seq_len=3"
        with self.assertRaisesRegex(ValueError, "Runtime flags"):
            comparison.compare_results(base, candidate, 2)

    def test_write_and_main(self):
        candidate = sample(label="candidate", sha="candidate-sha", tflops="11")
        compared = comparison.compare_results([sample()], [candidate], 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "nested" / "comparison.csv"
            comparison.write_comparison(output, compared)
            with output.open("r", encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

            base_path = root / "base.csv"
            candidate_path = root / "candidate.csv"
            fields = list(sample())
            for path, row in ((base_path, sample()), (candidate_path, candidate)):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(row)
            with mock.patch("builtins.print") as printed:
                self.assertEqual(
                    comparison.main([
                        "--base",
                        str(base_path),
                        "--candidate",
                        str(candidate_path),
                        "--expected-samples",
                        "1",
                        "--output",
                        str(output),
                    ]), 0)
            printed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
