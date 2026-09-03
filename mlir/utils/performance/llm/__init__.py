# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Ask a language model which perf configs to benchmark.

This package is the Python half of the `llm` and `llm-lfbo` tuning searches.
The C++ half (mlir/lib/Dialect/Rock/Tuning/LLMSearch.cpp) owns the round state
machine and everything that has to know about the tuning space; this half owns
the prompt, the model, and the salvaging of not-quite-JSON replies. They talk
over a JSON request and response, and `proposer.py` is the entry point.

The split, and most of the code, is ported from Helion's autotuner
(helion/autotuner/llm/ in pytorch/helion). Each module says at the top how
closely, since it varies a great deal: `parsing.py` is upstream's text
handling essentially unchanged, while `prompting.py` keeps upstream's
scaffolding and replaces every word of the Triton-specific advice inside it.
"""
