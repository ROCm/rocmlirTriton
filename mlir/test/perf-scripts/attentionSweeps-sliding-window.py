"""Pin the sliding-window invariants of ``attentionSweeps._sample_attn_shape``.

Runtime sliding-window masking is only valid in KV-cache mode: the
rock.attention verifier (and rocmlir-gen's ``sliding_window_size requires
current_seq_len`` guard, see rocmlir-gen/options.mlir) require a
``current_seqlen`` and a positive window that does not exceed ``seq_len_k``. If
the sweep ever emitted a sliding window without ``current_seqlen`` (or out of
range), every such sample would crash rocmlir-gen mid-sweep. This test locks
that contract, and that the sampled window round-trips into the perfRunner
``AttentionConfiguration`` tuning key (``-sliding_window_size`` present iff the
window is set, always paired with ``-current_seq_len``).

Doesn't need a GPU: only exercises the pure-Python shape sampler.

# RUN: %python %s | FileCheck %s
"""

import os
import shutil
import sys

# attentionSweeps.py is on PATH (lit's mlir_rock_tools_dir, populated by
# ci-performance-scripts). Resolve it and add its directory to sys.path so we
# can import the sampler as a module instead of duplicating it here.
_script = shutil.which('attentionSweeps.py')
if _script is None:
    sys.exit("attentionSweeps.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

import attentionSweeps  # noqa: E402

AttentionConfiguration = attentionSweeps.perfRunner.AttentionConfiguration

ARCH = "gfx950:sramecc+:xnack-"
NUM_CU = 256
NUM_CHIPLETS = 8
# Fixed seed keeps the run deterministic; 200 samples reliably covers both the
# sliding-window and plain KV-cache branches (~1/4 of samples set a window).
SAMPLES = 200
SEED = 0

ok = True


def fail(msg):
    global ok
    ok = False
    print(f"[MISMATCH] {msg}")


sliding_seen = 0
for shape, perf in attentionSweeps.random_attn_cases(SAMPLES, ARCH, seed=SEED):
    (dtype, g, slq, slk, nhq, nhkv, hdqk, hdv, scale, bias, tq, tk, tv, to, causal, return_lse,
     split_kv, current_seqlen, sliding_window_size) = shape

    if sliding_window_size < 0:
        fail(f"negative sliding_window_size {sliding_window_size}")
        continue

    if sliding_window_size == 0:
        continue

    sliding_seen += 1
    if current_seqlen is None:
        fail(f"sliding_window_size {sliding_window_size} without current_seqlen")
    if not (1 <= sliding_window_size <= slk):
        fail(f"sliding_window_size {sliding_window_size} outside [1, seq_len_k={slk}]")

    config = AttentionConfiguration(dtype=dtype,
                                    g=g,
                                    seq_len_q=slq,
                                    seq_len_k=slk,
                                    num_heads_q=nhq,
                                    num_heads_kv=nhkv,
                                    head_dim_qk=hdqk,
                                    head_dim_v=hdv,
                                    with_attn_scale=scale,
                                    with_attn_bias=bias,
                                    trans_q=tq,
                                    trans_k=tk,
                                    trans_v=tv,
                                    trans_o=to,
                                    causal=causal,
                                    return_lse=return_lse,
                                    split_kv=split_kv,
                                    arch=ARCH,
                                    num_cu=NUM_CU,
                                    num_chiplets=NUM_CHIPLETS,
                                    current_seqlen=current_seqlen,
                                    sliding_window_size=sliding_window_size)
    key = config.to_command_line()
    if f"-sliding_window_size {sliding_window_size}" not in key:
        fail(f"tuning key missing -sliding_window_size: {key}")
    if "-current_seq_len" not in key:
        fail(f"tuning key sets a window but omits -current_seq_len: {key}")

if sliding_seen == 0:
    fail(f"no sliding-window samples generated in {SAMPLES} samples; test is vacuous")

print(f"[OK] checked {SAMPLES} samples, {sliding_seen} with a sliding window")
# CHECK: [OK] checked

sys.exit(0 if ok else 1)
