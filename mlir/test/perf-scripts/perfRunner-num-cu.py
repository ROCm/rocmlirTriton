#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for querying the active HIP device through amd_arch_db.

# RUN: %python %s
"""

import contextlib
import io
import os
import shutil
import sys
import unittest

# (arch string, scheduling-unit count) per visible device.
DEVICES = [
    ("gfx942:sramecc+:xnack-", 304),
    ("gfx1170", 4),
]

# perfRunner.py depends on the compiled amd_arch_db binding. Both are deployed
# together by ci-performance-scripts, so import that copy rather than the source
# file, where the binding is unavailable.
_script = shutil.which("perfRunner.py")
if _script is None:
    sys.exit("perfRunner.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

# Keep this test independent of the host GPU while exercising the pybind calls
# perfRunner makes. perfRunner queries the device only through these, so there
# is no HIP module left to stub out.
import amd_arch_db  # noqa: E402

_visible = list(DEVICES)
amd_arch_db.get_native_device_count = lambda: len(_visible)
amd_arch_db.get_native_arch = lambda device_id=0: (_visible[device_id][0]
                                                   if device_id < len(_visible) else None)
amd_arch_db.get_native_num_cu = lambda chip: next(
    (numCU for arch, numCU in _visible if chip in arch), None)

# Start in CU mode so importing perfRunner must warn and force WGP mode.
os.environ["GPU_ENABLE_WGP_MODE"] = "0"
IMPORT_STDERR = io.StringIO()
with contextlib.redirect_stderr(IMPORT_STDERR):
    from perfRunner import get_arch, get_num_cu  # noqa: E402


class GetNumCuTest(unittest.TestCase):

    def test_forces_wgp_mode(self):
        self.assertEqual(os.environ["GPU_ENABLE_WGP_MODE"], "1")

    def test_warns_before_overriding_cu_mode(self):
        self.assertIn("GPU_ENABLE_WGP_MODE=0 is overridden to 1", IMPORT_STDERR.getvalue())

    def test_returns_matching_device_multiprocessor_count(self):
        self.assertEqual(get_num_cu("gfx1170"), 4)
        self.assertEqual(get_num_cu("gfx942"), 304)

    def test_rejects_unknown_chip(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot find number of CUs for gfx9999"):
            get_num_cu("gfx9999")


class GetArchTest(unittest.TestCase):
    """get_arch enumerates devices through the binding, not through HIP."""

    def setUp(self):
        self._saved = list(_visible)

    def tearDown(self):
        _visible[:] = self._saved

    def test_returns_the_only_agent(self):
        _visible[:] = [DEVICES[0], DEVICES[0]]
        self.assertEqual(get_arch(), DEVICES[0][0])

    def test_warns_on_mixed_agents(self):
        _visible[:] = list(DEVICES)
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stderr):
            arch = get_arch()
        self.assertIn("different kinds of agents", stderr.getvalue())
        self.assertIn(arch, [arch for arch, _ in DEVICES])


if __name__ == "__main__":
    unittest.main()
