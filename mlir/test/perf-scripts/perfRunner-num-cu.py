#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for querying the active HIP scheduling-unit count.

# RUN: %python %s
"""

from enum import IntEnum
import os
from pathlib import Path
import sys
import types
import unittest


class HipError(IntEnum):
    hipSuccess = 0  # noqa: N815 - Match the HIP Python API spelling.


DEVICES = [
    (b"gfx942:sramecc+:xnack-", 304),
    (b"gfx1170", 4),
]


class HipDeviceProperties:

    def __init__(self):
        self.gcnArchName = b""
        self.multiProcessorCount = 0


def get_device_count():
    return HipError.hipSuccess, len(DEVICES)


def get_device_properties(props, device):
    props.gcnArchName, props.multiProcessorCount = DEVICES[device]
    return HipError.hipSuccess,


mock_hip = types.SimpleNamespace(hipError_t=HipError,
                                 hipDeviceProp_t=HipDeviceProperties,
                                 hipGetDeviceCount=get_device_count,
                                 hipGetDeviceProperties=get_device_properties)
hip_package = types.ModuleType("hip")
hip_package.hip = mock_hip
sys.modules["hip"] = hip_package

MLIR_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MLIR_DIR / "utils" / "performance"))

os.environ.pop("GPU_ENABLE_WGP_MODE", None)
from perfRunner import get_num_cu  # noqa: E402


class GetNumCuTest(unittest.TestCase):

    def test_forces_wgp_mode(self):
        self.assertEqual(os.environ["GPU_ENABLE_WGP_MODE"], "1")

    def test_returns_matching_device_multiprocessor_count(self):
        self.assertEqual(get_num_cu("gfx1170"), 4)
        self.assertEqual(get_num_cu("gfx942"), 304)

    def test_rejects_unknown_chip(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot find number of CUs for gfx9999"):
            get_num_cu("gfx9999")


if __name__ == "__main__":
    unittest.main()
