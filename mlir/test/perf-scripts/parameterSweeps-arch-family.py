# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Pin ``parameterSweeps._arch_family`` and the two per-arch knobs it drives
(``_compile_cost_budget`` and ``_timeout_rate``), plus the fp8
``_dtype_amplifier``, which fires on the WMMA-era RDNA line only.

``_arch_family`` buckets an arch as ``'rdna'`` (the consumer line, via
``rock::isRDNA`` through the AmdArchDB pybind module) or ``'cdna'`` (the
catch-all). Each case also pins ``amd_arch_db.is_cdna`` / ``is_rdna``
themselves, so a binding wired to the wrong ``rock::`` predicate is caught
here and not by a silently resized budget. The two are not complements, so
the cases below cover all three groups Triton distinguishes: CDNA proper,
RDNA proper, and the archs in neither switch (gfx906). The interesting
entries are the ones a gfx-number range gets wrong:

  * gfx1250 is a gfx12 chip but a CDNA part, so it takes the looser budget.
  * gfx1170 is its own ISA family but still RDNA, so it takes the tighter one.

Doesn't need a GPU: only exercises the pure-Python classifiers.

# RUN: %python %s | FileCheck %s
"""

import os
import shutil
import sys

# parameterSweeps.py is on PATH (lit's mlir_rock_tools_dir, populated by
# ci-performance-scripts). Resolve it and add its directory to sys.path so
# we can import the classifiers as a module instead of duplicating them here.
_script = shutil.which('parameterSweeps.py')
if _script is None:
    sys.exit("parameterSweeps.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

import parameterSweeps  # noqa: E402

# Dropped next to the scripts by ci-performance-scripts, so the same sys.path
# entry finds it.
import amd_arch_db  # noqa: E402

# (label, arch, expected_family, expected_is_cdna). ``expected_family`` also
# pins the budget and the per-operation timeout rates, which are looked up by
# family, and pins ``is_rdna`` since the two agree by construction.
# ``expected_is_cdna`` is carried separately precisely because it does NOT
# follow from the family: gfx906 buckets as ``'cdna'`` yet ``is_cdna`` is
# False, which is the whole reason the bucket is keyed on ``is_rdna``.
FAMILY_CASES = [
    ("gcn5_1", "gfx906", 'cdna', False),  # in neither isCDNA nor isRDNA
    ("cdna1", "gfx908", 'cdna', True),
    ("cdna2", "gfx90a", 'cdna', True),
    ("cdna3", "gfx942", 'cdna', True),
    ("cdna4", "gfx950", 'cdna', True),
    ("rdna1", "gfx1010", 'rdna', False),
    ("rdna2", "gfx1030", 'rdna', False),
    ("rdna3", "gfx1100", 'rdna', False),
    ("gfx1170", "gfx1170", 'rdna', False),
    ("rdna4", "gfx1201", 'rdna', False),
    ("gfx1250", "gfx1250", 'cdna', True),
    # Decorated arch strings: HIP gcnArchName and the LLVM triple.
    ("cdna4 hip", "gfx950:sramecc+:xnack-", 'cdna', True),
    ("rdna4 llvm-triple", "amdgcn-amd-amdhsa:gfx1201", 'rdna', False),
]

# (label, dtype, arch, expected_alpha). fp8 is amplified on the WMMA-era RDNA
# line only, which the amplifier spells as "RDNA minus RDNA1/RDNA2" so a future
# RDNA family needs no edit. gfx1030 pins that subtraction; gfx1250 pins the
# other edge, since it runs WMMA but is CDNA and so stays at 1.
AMPLIFIER_CASES = [
    ("f16 anywhere", "f16", "gfx1100", 1),
    ("fp8 cdna3", "fp8", "gfx942", 1),
    ("fp8 cdna4", "fp8", "gfx950", 1),
    ("fp8 rdna1", "fp8", "gfx1010", 1),
    ("fp8 rdna2", "fp8", "gfx1030", 1),
    ("fp8 rdna3", "fp8", "gfx1100", 10),
    ("fp8 gfx1170", "fp8", "gfx1170", 10),
    ("fp8 rdna4", "fp8", "gfx1201", 10),
    ("fp8 gfx1250", "fp8", "gfx1250", 1),
]

# Budget and timeout rate per family, keyed off _arch_family's answer.
EXPECTED_BUDGET = {'cdna': 12000, 'rdna': 8000}
EXPECTED_ATTN_TIMEOUT_RATE = {'cdna': 0.035, 'rdna': 0.045}

ok = True
for label, arch, expected, expected_is_cdna in FAMILY_CASES:
    got = parameterSweeps._arch_family(arch)
    budget = parameterSweeps._compile_cost_budget(arch)
    rate = parameterSweeps._timeout_rate(arch, 'attn')
    # Straight through the pybind, so a binding wired to the wrong rock::
    # predicate fails here rather than silently resizing every budget.
    is_cdna = amd_arch_db.is_cdna(arch)
    is_rdna = amd_arch_db.is_rdna(arch)
    matches = (got == expected and budget == EXPECTED_BUDGET[expected] and
               rate == EXPECTED_ATTN_TIMEOUT_RATE[expected] and is_cdna == expected_is_cdna and
               is_rdna == (expected == 'rdna'))
    if not matches:
        ok = False
    status = "OK" if matches else "MISMATCH"
    print(f"[{status}] {label:18} arch={arch!r:34} expected={expected} got={got} "
          f"is_cdna={is_cdna} is_rdna={is_rdna} budget={budget} attn_rate={rate}")

for label, dtype, arch, expected in AMPLIFIER_CASES:
    got = parameterSweeps._dtype_amplifier(dtype, arch)
    status = "OK" if got == expected else "MISMATCH"
    if got != expected:
        ok = False
    print(f"[{status}] {label:18} arch={arch!r:34} expected=alpha{expected} got=alpha{got}")

# CHECK lines run in order, so they also pin the iteration order of the cases.
# CHECK: [OK] gcn5_1
# CHECK: [OK] cdna1
# CHECK: [OK] cdna2
# CHECK: [OK] cdna3
# CHECK: [OK] cdna4
# CHECK: [OK] rdna1
# CHECK: [OK] rdna2
# CHECK: [OK] rdna3
# CHECK: [OK] gfx1170
# CHECK: [OK] rdna4
# CHECK: [OK] gfx1250
# CHECK: [OK] cdna4 hip
# CHECK: [OK] rdna4 llvm-triple
# CHECK: [OK] f16 anywhere
# CHECK: [OK] fp8 cdna3
# CHECK: [OK] fp8 cdna4
# CHECK: [OK] fp8 rdna1
# CHECK: [OK] fp8 rdna2
# CHECK: [OK] fp8 rdna3
# CHECK: [OK] fp8 gfx1170
# CHECK: [OK] fp8 rdna4
# CHECK: [OK] fp8 gfx1250

sys.exit(0 if ok else 1)
