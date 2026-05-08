# gemm+gemm companion to ``parameterSweeps-conv.py``: shares the attn:v1:
# perf-config family with attention but drives the GemmGemmConfiguration
# problem space (independent split-K, no causal mask, no KV-cache).
#
# RUN: attentionSweeps.py gemm_gemm --samples 1 --seed 0 --jobs 1 --quiet \
# RUN:     --test-timeout-sec 20 > %t 2>&1 || true
# RUN: FileCheck %s < %t
#
# CHECK: Passed: {{[0-9]+}}, Not applicable: {{[0-9]+}}, Failed: {{[0-9]+}}
