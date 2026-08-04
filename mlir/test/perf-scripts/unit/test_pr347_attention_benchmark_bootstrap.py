# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %python %s

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[3] / "utils" / "performance" /
          "runPR347AttentionBenchmark.sh")


class PR347AttentionBenchmarkBootstrapTest(unittest.TestCase):

    @staticmethod
    def run_command(command, cwd=None, env=None):
        return subprocess.run(command, cwd=cwd, env=env, check=True, capture_output=True, text=True)

    @staticmethod
    def write_executable(path, contents):
        path.write_text(contents, encoding="utf-8")
        path.chmod(0o755)

    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_help(self):
        completed = subprocess.run(["bash", str(SCRIPT), "--help"],
                                   check=True,
                                   capture_output=True,
                                   text=True)
        self.assertIn("Fetch, build, quick-tune, and benchmark", completed.stdout)
        self.assertIn("--workspace PATH", completed.stdout)

    def test_gpu_free_end_to_end_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "origin.git"
            seed = root / "seed"
            clone = root / "clone"
            workspace = root / "workspace"
            fake_bin = root / "bin"
            fake_bin.mkdir()

            self.run_command(["git", "init", "--bare", str(remote)])
            self.run_command(["git", "init", str(seed)])
            self.run_command(["git", "remote", "add", "origin", str(remote)], cwd=seed)

            self.write_executable(
                seed / "cmake.sh", """#!/usr/bin/env bash
set -eu
mkdir -p build/bin
printf 'CMAKE_HOME_DIRECTORY:INTERNAL=%s\\n' "$PWD" > build/CMakeCache.txt
touch build/build.ninja
touch build/bin/rocmlir-gen build/bin/rocmlir-driver build/bin/rocmlir-tuning-driver
""")
            (seed / ".gitignore").write_text("/build/\n", encoding="utf-8")
            self.run_command(["git", "add", "cmake.sh", ".gitignore"], cwd=seed)
            self.run_command([
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m",
                "base"
            ],
                             cwd=seed)
            base_sha = self.run_command(["git", "rev-parse", "HEAD"], cwd=seed).stdout.strip()

            performance = seed / "mlir" / "utils" / "performance"
            configs = performance / "configs"
            configs.mkdir(parents=True)
            (configs / "tier1-attention-configs").write_text("-t f16\n", encoding="utf-8")
            (performance / "filterAttentionConfigs.py").write_text("""import json
import sys
from pathlib import Path
output = Path(sys.argv[sys.argv.index("--output") + 1])
output.write_text("-t f16 -g 1 -seq_len_q 1 -seq_len_k 2\\n")
output.with_suffix(output.suffix + ".json").write_text(json.dumps({"count": 1}))
""",
                                                                   encoding="utf-8")
            (performance / "runAttentionBranchBenchmark.py").write_text("""import sys
from pathlib import Path
output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
output.mkdir(parents=True, exist_ok=True)
(output / "invoked").write_text("ok\\n")
""",
                                                                        encoding="utf-8")
            self.run_command(["git", "add", "mlir"], cwd=seed)
            self.run_command([
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m",
                "candidate"
            ],
                             cwd=seed)
            candidate_sha = self.run_command(["git", "rev-parse", "HEAD"], cwd=seed).stdout.strip()
            self.run_command(["git", "branch", "users/umayadav/pr347-attention-perf"], cwd=seed)
            self.run_command(["git", "push", "origin", "users/umayadav/pr347-attention-perf"],
                             cwd=seed)
            self.run_command([
                "git", "--git-dir",
                str(remote), "symbolic-ref", "HEAD",
                "refs/heads/users/umayadav/pr347-attention-perf"
            ])
            self.run_command(["git", "clone", "--depth", "1", f"file://{remote}", str(clone)])
            self.assertEqual(
                self.run_command(["git", "rev-parse", "--is-shallow-repository"],
                                 cwd=clone).stdout.strip(), "true")
            missing_base = subprocess.run(["git", "cat-file", "-e", f"{base_sha}^{{commit}}"],
                                          cwd=clone,
                                          capture_output=True)
            self.assertNotEqual(missing_base.returncode, 0)

            for command in ("cmake", "ninja", "rocm-smi"):
                self.write_executable(fake_bin / command, "#!/usr/bin/env bash\nexit 0\n")
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

            completed = self.run_command([
                "bash",
                str(SCRIPT), "--repo",
                str(clone), "--workspace",
                str(workspace), "--base-ref", base_sha, "--candidate-sha", candidate_sha,
                "--foreground"
            ],
                                         env=environment)
            self.assertIn("Launching PR #347 attention benchmark", completed.stdout)
            self.assertTrue((workspace / "results" / "invoked").is_file())
            self.assertTrue(
                (workspace / "configs" / "pr347-causal-kvcache-attention-configs").is_file())

            # A cache left by an interrupted configure is recoverable.
            (workspace / "sources" / "base" / "build" / "build.ninja").unlink()
            self.run_command([
                "bash",
                str(SCRIPT), "--repo",
                str(clone), "--workspace",
                str(workspace), "--base-ref", base_sha, "--candidate-sha", candidate_sha,
                "--foreground"
            ],
                             env=environment)


if __name__ == "__main__":
    unittest.main()
