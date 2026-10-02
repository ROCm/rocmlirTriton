#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for createFusionPerformanceReports.py.

perfRunner records one fusion result per test file. Two files can fuse
different operations around the same conv or GEMM problem, so their rows share
every problem column, and the report has to render them as separate rows. The
problems and file names below come from a gfx942 run over resnet50-e2e and
bert-torch-tosa-e2e.

# RUN: %python %s %t
"""

import os
import shutil
import sys
import unittest
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parents[2] / "utils" / "performance"
sys.path.insert(0, str(PERF_DIR))

import createFusionPerformanceReports  # noqa: E402
import reportUtils  # noqa: E402

TMP_PREFIX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/fusion-performance-report")
CHIP = "gfx942"

CONV_CSV = """\
Direction,DataType,Chip,numCU,numChiplets,FilterLayout,InputLayout,OutputLayout,N,C,H,W,K,Y,X,DilationH,DilationW,StrideH,StrideW,PaddingH,PaddingW,PerfConfig,LDSBankConflict,Fusion TFlops,MLIR TFlops,Fusion/MLIR,FileName
fwd,i8,gfx942,304,8,gkc01,ngc01,ngk01,1,128,56,56,128,3,3,1,1,2,2,1,1,,NaN,0.712234,1.522267,0.467877,/workspace/mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1-quantization.mlir
fwd,i8,gfx942,304,8,gkc01,ngc01,ngk01,1,128,56,56,128,3,3,1,1,2,2,1,1,,NaN,0.730365,1.522106,0.479839,/workspace/mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1-int8.mlir
fwd,f32,gfx942,304,8,gkc01,ngc01,ngk01,1,384,28,28,512,1,1,1,1,1,1,0,0,,NaN,21.394511,37.900337,0.564495,/workspace/mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-15.mlir
"""

GEMM_CSV = """\
DataType,OutDataType,Chip,numCU,numChiplets,TransA,TransB,TransO,G,M,K,N,ScaledGemm,ScaleADtype,ScaleBDtype,TransScaleA,TransScaleB,PerfConfig,LDSBankConflict,Fusion TFlops,MLIR TFlops,Fusion/MLIR,FileName
f32,f32,gfx942,304,8,False,False,False,1,12,384,384,False,,,False,False,,NaN,0.437015,0.695274,0.628550,/workspace/mlir/test/xmir/bert-torch-tosa-e2e/bert_part_5.torch-tosa.mlir
f32,f32,gfx942,304,8,False,False,False,1,12,384,384,False,,,False,False,,NaN,0.490362,0.664466,0.737980,/workspace/mlir/test/xmir/bert-torch-tosa-e2e/bert_part_0.torch-tosa.mlir
f32,f32,gfx942,304,8,False,False,False,1,12,1536,384,False,,,False,False,,NaN,0.912345,1.033948,0.882390,/workspace/mlir/test/xmir/bert-torch-tosa-e2e/bert_part_7.torch-tosa.mlir
"""


class FusionReportSharedProblemTest(unittest.TestCase):
    """Fusions of the same problem from different files stay separate rows."""

    def setUp(self):
        self.work_dir = Path(f"{TMP_PREFIX}.{self._testMethodName}")
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.work_dir)

    def render(self, op, csv):
        Path(f"{CHIP}_{op}_{reportUtils.PERF_REPORT_FUSION_FILE}").write_text(csv)
        createFusionPerformanceReports.print_all_performance(CHIP, op)
        return Path(f"{CHIP}_{op}_fusion.html").read_text()

    def test_conv_problem_fused_in_two_files(self):
        html = self.render("conv", CONV_CSV)
        self.assertIn("mixr-resnet-fusion-case-1-quantization.mlir", html)
        self.assertIn("mixr-resnet-fusion-case-1-int8.mlir", html)

    def test_gemm_problem_fused_in_two_files(self):
        html = self.render("gemm", GEMM_CSV)
        self.assertIn("bert_part_5.torch-tosa.mlir", html)
        self.assertIn("bert_part_0.torch-tosa.mlir", html)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
