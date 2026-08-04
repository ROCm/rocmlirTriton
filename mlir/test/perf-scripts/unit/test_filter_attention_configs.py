# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: env PYTHONPATH=%S/../../../utils/performance %python %s

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import attentionPerfUtils
import filterAttentionConfigs

utils = attentionPerfUtils
filtering = filterAttentionConfigs


class AttentionPerfUtilsTest(unittest.TestCase):

    def test_parse_and_canonicalize(self):
        self.assertEqual(utils.normalize_option("--with-attn-scale"), "with_attn_scale")
        self.assertEqual(utils.parse_config("-g 2 --causal=true"), {"g": "2", "causal": "true"})
        self.assertEqual(utils.canonical_config("--causal=true -g 2"), "causal=true g=2")
        with self.assertRaisesRegex(ValueError, "Expected an option"):
            utils.parse_config("attention")
        with self.assertRaisesRegex(ValueError, "Missing value"):
            utils.parse_config("-g")
        with self.assertRaisesRegex(ValueError, "Duplicate option"):
            utils.parse_config("-g 1 --g=2")

    def test_boolean_and_option_helpers(self):
        for value in ("1", "true", "TRUE"):
            self.assertTrue(utils.parse_bool(value))
        for value in ("0", "false", "FALSE"):
            self.assertFalse(utils.parse_bool(value))
        with self.assertRaisesRegex(ValueError, "Invalid boolean"):
            utils.parse_bool("maybe")
        self.assertEqual(utils.option_value({"seq_len_q": "1"}, "--seq-len-q"), "1")
        self.assertEqual(utils.option_value({}, "missing", "fallback"), "fallback")
        self.assertEqual(utils.remove_flag(["--detach", "--gpu", "2"], "--detach"), ["--gpu", "2"])

    def test_file_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "configs"
            source.write_text("\n# comment\n-g 1\n -g 2 \n", encoding="utf-8")
            self.assertEqual(utils.read_config_lines(source), [(3, "-g 1"), (4, "-g 2")])

            lines = root / "lines"
            utils.write_lines(lines, ["one", "two"])
            self.assertEqual(lines.read_text(encoding="utf-8"), "one\ntwo\n")
            utils.write_lines(lines, [])
            self.assertEqual(lines.read_text(encoding="utf-8"), "")

            data = root / "nested" / "data.json"
            utils.write_json(data, {"b": 2, "a": 1})
            self.assertEqual(json.loads(data.read_text(encoding="utf-8")), {"a": 1, "b": 2})

    def test_atomic_write_cleanup_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "output"
            with mock.patch.object(os, "replace", side_effect=OSError("replace")):
                with self.assertRaisesRegex(OSError, "replace"):
                    utils.atomic_write_text(destination, "value")
            self.assertEqual(list(Path(directory).iterdir()), [])

            with mock.patch.object(os, "replace", side_effect=OSError("replace")), \
                    mock.patch.object(os, "unlink", side_effect=FileNotFoundError):
                with self.assertRaisesRegex(OSError, "replace"):
                    utils.atomic_write_text(destination, "value")


class FilterAttentionConfigsTest(unittest.TestCase):

    def test_classification(self):
        definite = [
            "-causal true -seq_len_q 4",
            "-prefix_offset 3 -seq_len_q 4",
            "-current_seq_len 3 -seq_len_q 1",
        ]
        expected_reasons = ["causal", "prefix-causal", "kv-cache"]
        for config, reason in zip(definite, expected_reasons):
            confidence, reasons = filtering.classify_config(config)
            self.assertEqual(confidence, "definite")
            self.assertIn(reason, reasons)

        self.assertEqual(filtering.classify_config("-seq_len_q 1"),
                         ("potential", ["decode-shaped SeqLenQ=1 KV-cache proxy"]))
        self.assertEqual(filtering.classify_config("-seq_len_q 1", conservative=False), (None, []))
        self.assertEqual(filtering.classify_config("-seq_len_q 8"), (None, []))
        with self.assertRaises(ValueError):
            filtering.classify_config("-causal invalid")

    def test_filter_and_main(self):
        selected, records = filtering.filter_configs([
            (1, "-causal true -seq_len_q 4"),
            (2, "-seq_len_q 1"),
            (3, "-seq_len_q 8"),
        ])
        self.assertEqual(len(selected), 2)
        self.assertEqual([record["confidence"] for record in records], ["definite", "potential"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            output = root / "output"
            manifest = root / "manifest.json"
            source.write_text("-causal true -seq_len_q 4\n-seq_len_q 1\n", encoding="utf-8")
            with mock.patch("builtins.print") as output_mock:
                self.assertEqual(
                    filtering.main([
                        "--input",
                        str(source),
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                        "--mode",
                        "exact",
                    ]), 0)
            output_mock.assert_called_once()
            self.assertEqual(output.read_text(encoding="utf-8"), "-causal true -seq_len_q 4\n")
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["selected_count"], 1)

            default_manifest = Path(f"{output}.json")
            filtering.main(["--input", str(source), "--output", str(output)])
            self.assertTrue(default_manifest.exists())

    def test_committed_filter_artifact_is_current(self):
        repository = Path(__file__).resolve().parents[4]
        source = repository / "mlir" / "utils" / "performance" / "configs" / \
            "tier1-attention-configs"
        artifact = source.parent / "pr347-impacted-tier1-attention-configs"
        selected, records = filtering.filter_configs(utils.read_config_lines(source))
        self.assertEqual(artifact.read_text(encoding="utf-8"), "\n".join(selected) + "\n")
        self.assertEqual(sum(record["confidence"] == "definite" for record in records), 2)
        self.assertEqual(sum(record["confidence"] == "potential" for record in records), 2)
        manifest = json.loads(Path(f"{artifact}.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest, {
                "source": "mlir/utils/performance/configs/tier1-attention-configs",
                "mode": "conservative",
                "input_count": len(utils.read_config_lines(source)),
                "selected_count": len(selected),
                "definite_count": 2,
                "potential_count": 2,
                "selected": records,
            })


if __name__ == "__main__":
    unittest.main()
