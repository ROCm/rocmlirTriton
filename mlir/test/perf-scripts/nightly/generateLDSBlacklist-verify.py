# Drift check for the compiled-in LDS blacklist (LdsBlacklistPerfconfigs.inc).
#
# Re-lowers a random sample of the perf configs currently recorded in the
# in-tree .inc and confirms each still overflows LDS with today's Triton/
# backend. If a config no longer overflows (e.g. after a Triton bump changed
# shared-memory usage), the script exits non-zero and this test fails,
# signalling the .inc must be regenerated with generateLDSBlacklist.py.
# Compile-only, so no GPU is needed; it runs in nightly (see lit.local.cfg).
#
# We sample only some entries (verifying the full ~20k-entry table every night is
# too slow) with the seed defaulting to the ISO week, so a given week checks a
# stable subset and successive weeks rotate through the table. An empty .inc is
# treated as a failure (it means the table was lost or never generated), so the
# verify must report that entries still overflow LDS to pass.
#
# RUN: generateLDSBlacklist.py --verify --samples 2000 \
# RUN:     --output %mlir_src_root/include/mlir/Dialect/Rock/Tuning/LdsBlacklistPerfconfigs.inc \
# RUN:     2>&1 | FileCheck %s
#
# CHECK: still overflow LDS
