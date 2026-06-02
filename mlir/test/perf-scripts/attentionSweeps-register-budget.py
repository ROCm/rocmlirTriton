"""Pin ``attentionSweeps._waves_per_eu_register_budget_ok`` across every
(wave_size, vgprs_per_eu) regime the predicate distinguishes:

  (wave_size, vgprs_per_eu)   archs                                 budget/thr at wpe=8
  ---------------------------------------------------------------------------
  (64, 512)                   gfx9xx pre-1100 (CDNA1-4)             64
  (32, 512)                   gfx10xx (RDNA1/2)                     64
  (32, 1024)                  gfx11xx, gfx12 < 1250 (RDNA3/4)       128
  (64, 1024)                  gfx1250+ and (presumed) gfx13+        128
  (64, 512) fallback          unknown / unparseable archs           64

Each regime gets a just-fits ACCEPT and a just-overflows REJECT pair so a
future calibration tweak fails fast. ``wavesPerEU == 0`` (treated as
``wpe == 1``) and the multi-wave path are exercised on multiple archs.
Both arch-string forms ``_arch_id`` accepts are covered: HIP gcnArchName
(``gfx950:sramecc+:xnack-``) and LLVM-triple
(``amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-``).

Doesn't need a GPU: only exercises the pure-Python predicate.

# RUN: %python %s | FileCheck %s
"""

import os
import shutil
import sys

# attentionSweeps.py is on PATH (lit's mlir_rock_tools_dir, populated by
# ci-performance-scripts). Resolve it and add its directory to sys.path so
# we can import the predicate as a module instead of duplicating it here.
_script = shutil.which('attentionSweeps.py')
if _script is None:
    sys.exit("attentionSweeps.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

import attentionSweeps  # noqa: E402

# (label, arch, perf_config, expected). perf_config follows the 11-field
# tuple parsed by GemmGemmParamsAttr / GemmParamsAttr: (mPerBlock, nPerBlock,
# kPerBlock, kpack, numCTAs, numWaves, matrixInstrNonkdim, splitKFactor,
# numStages, wavesPerEU, gridGroupSize).
GFX950_HIP = "gfx950:sramecc+:xnack-"
GFX950_LLVM = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-"

CASES = [
    # gfx950 user configs (CDNA4, wave=64, vgprs=512): wpe=8 budget=64.
    # (256*32)/64 = 128 > 64 -> REJECT.
    ("cfg1 bf16", GFX950_HIP, (256, 32, 16, 1, 1, 1, 0, 1, 2, 8, 1), False),
    # wpe=0 -> eff_wpe=1, budget=512. (256*128)/64 = 512 fits exactly -> ACCEPT.
    ("cfg2 i8", GFX950_HIP, (256, 128, 64, 1, 1, 1, 16, 1, 1, 0, 8), True),
    # (256*128)/64 = 512 >> 64 -> REJECT.
    ("cfg3 f16", GFX950_HIP, (256, 128, 16, 1, 1, 1, 16, 1, 3, 8, 0), False),
    # (32*256)/64 = 128 > 64 -> REJECT.
    ("cfg4 i8", GFX950_HIP, (32, 256, 16, 2, 1, 1, 16, 1, 2, 8, 8), False),
    # cfg1 via the LLVM-triple form -- exercises that branch of _arch_id.
    ("cfg1 llvm-triple", GFX950_LLVM, (256, 32, 16, 1, 1, 1, 0, 1, 2, 8, 1), False),

    # gfx942 (CDNA3): same regime as gfx950.
    ("gfx942 fits", "gfx942", (64, 64, 16, 1, 1, 1, 16, 1, 1, 8, 0), True),
    ("gfx942 overflow", "gfx942", (128, 64, 16, 1, 1, 1, 16, 1, 1, 8, 0), False),

    # gfx1030 (RDNA2, wave=32, vgprs=512): half wave -> half threads -> half budget.
    # (64*32)/32 = 64 -> ACCEPT; (64*64)/32 = 128 -> REJECT.
    ("gfx1030 fits", "gfx1030", (64, 32, 16, 1, 1, 1, 16, 1, 1, 8, 0), True),
    ("gfx1030 overflow", "gfx1030", (64, 64, 16, 1, 1, 1, 16, 1, 1, 8, 0), False),

    # gfx1100 (RDNA3, wave=32, vgprs=1024): doubled VGPR file -> wpe=8 budget=128.
    # (64*64)/32 = 128 -> ACCEPT; (128*64)/32 = 256 -> REJECT.
    ("gfx1100 fits", "gfx1100", (64, 64, 16, 1, 1, 1, 16, 1, 1, 8, 0), True),
    ("gfx1100 overflow", "gfx1100", (128, 64, 16, 1, 1, 1, 16, 1, 1, 8, 0), False),

    # gfx1201 (RDNA4): same regime as gfx1100.
    ("gfx1201 fits", "gfx1201", (64, 64, 16, 1, 1, 1, 16, 1, 1, 8, 0), True),
    ("gfx1201 overflow", "gfx1201", (128, 64, 16, 1, 1, 1, 16, 1, 1, 8, 0), False),

    # gfx1250 (wave=64, vgprs=1024): same wave as gfx950 but doubled file.
    # (128*64)/64 = 128 -> ACCEPT; (128*128)/64 = 256 -> REJECT.
    ("gfx1250 fits", "gfx1250", (128, 64, 16, 1, 1, 1, 16, 1, 1, 8, 0), True),
    ("gfx1250 overflow", "gfx1250", (128, 128, 16, 1, 1, 1, 16, 1, 1, 8, 0), False),

    # Unknown arch falls through to (wave=64, vgprs=512); locks the
    # conservative default so a future _arch_id refactor can't silently
    # widen the unknown bucket.
    ("unknown fits", "made-up-target", (64, 64, 16, 1, 1, 1, 16, 1, 1, 8, 0), True),
    ("unknown overflow", "made-up-target", (128, 64, 16, 1, 1, 1, 16, 1, 1, 8, 0), False),

    # wpe=0 -> eff_wpe=1, budget=vgprs_per_eu. C-tiles wider than the full
    # SIMD VGPR file are rejected on every arch.
    # (256*256)/64 = 1024 > 512 -> REJECT (gfx950 / unknown).
    ("wpe=0 1024-tile gfx950", GFX950_HIP, (256, 256, 16, 1, 1, 1, 16, 1, 1, 0, 0), False),
    ("wpe=0 1024-tile unknown", "made-up-target", (256, 256, 16, 1, 1, 1, 16, 1, 1, 0, 0), False),
    # (256*256)/32 = 2048 > 1024 -> REJECT (gfx1100, doubled file).
    ("wpe=0 2048-tile gfx1100", "gfx1100", (256, 256, 16, 1, 1, 1, 16, 1, 1, 0, 0), False),
    # Companion ACCEPTs: small tile, and bumping num_waves -> small accum/thread.
    ("wpe=0 small gfx950", GFX950_HIP, (64, 64, 16, 1, 1, 1, 16, 1, 1, 0, 0), True),
    # (256*256)/(8*64) = 128 <= 512 -> ACCEPT.
    ("wpe=0 multi-wave gfx950", GFX950_HIP, (256, 256, 16, 1, 1, 8, 16, 1, 1, 0, 0), True),

    # Multi-wave: num_waves=4 quadruples threads -> accum/thread / 4.
    # (128*128)/(4*64) = 64 fits gfx950 wpe=8 budget=64.
    ("4-wave fits", GFX950_HIP, (128, 128, 16, 1, 1, 4, 16, 1, 1, 8, 0), True),
]

ok = True
for label, arch, perf_config, expected in CASES:
    got = attentionSweeps._waves_per_eu_register_budget_ok(perf_config, arch)
    verdict = "ACCEPT" if got else "REJECT"
    expected_str = "ACCEPT" if expected else "REJECT"
    status = "OK" if got == expected else "MISMATCH"
    if got != expected:
        ok = False
    # Column-aligned so a partial failure is easy to eyeball next to OK lines.
    print(f"[{status}] {label:22} arch={arch!r:42} expected={expected_str} got={verdict}")

# CHECK lines run in order, so they also pin the iteration order of CASES.
# CHECK: [OK] cfg1 bf16
# CHECK: [OK] cfg2 i8
# CHECK: [OK] cfg3 f16
# CHECK: [OK] cfg4 i8
# CHECK: [OK] cfg1 llvm-triple
# CHECK: [OK] gfx942 fits
# CHECK: [OK] gfx942 overflow
# CHECK: [OK] gfx1030 fits
# CHECK: [OK] gfx1030 overflow
# CHECK: [OK] gfx1100 fits
# CHECK: [OK] gfx1100 overflow
# CHECK: [OK] gfx1201 fits
# CHECK: [OK] gfx1201 overflow
# CHECK: [OK] gfx1250 fits
# CHECK: [OK] gfx1250 overflow
# CHECK: [OK] unknown fits
# CHECK: [OK] unknown overflow
# CHECK: [OK] wpe=0 1024-tile gfx950
# CHECK: [OK] wpe=0 1024-tile unknown
# CHECK: [OK] wpe=0 2048-tile gfx1100
# CHECK: [OK] wpe=0 small gfx950
# CHECK: [OK] wpe=0 multi-wave gfx950
# CHECK: [OK] 4-wave fits

sys.exit(0 if ok else 1)
