# Attention companion to ``parameterSweeps-conv.py``: same harness, also
# exercises the host-highlevel pre-pipeline (tosa.* lowering for ``-pv``)
# that gemm/conv don't hit.
#
# RUN: attentionSweeps.py attention --samples 1 --seed 0 --jobs 1 --quiet \
# RUN:     --test-timeout-sec 20 > %t 2>&1 || true
# RUN: FileCheck %s < %t
#
# CHECK: Passed: {{[0-9]+}}, Not applicable: {{[0-9]+}}, Failed: {{[0-9]+}}
