#!/bin/bash
# Add RUN and CHECK lines to files that only have a TODO comment

CONTENT='// TODO(roctriton): We need to unbufferize rock.reduce
// RUN: rocmlir-opt %s | FileCheck %s
// CHECK: module
'

FILES=(
  "integration/reduce/reduce_max/rock-reduce-max-case1.mlir"
  "integration/reduce/reduce_max/rock-reduce-max-case2.mlir"
  "integration/reduce/reduce_max/rock-reduce-max-case3.mlir"
  "integration/reduce/reduce_max/rock-reduce-max-case4.mlir"
  "integration/reduce/blockwise_reduce/blockwise_reducemax_nr_threads_gt_blocksize.mlir"
  "integration/reduce/blockwise_reduce/blockwise_reducemax_nr_threads_lt_blocksize.mlir"
  "integration/reduce/blockwise_reduce/blockwise_reducesum_nr_threads_gt_blocksize.mlir"
  "integration/reduce/blockwise_reduce/blockwise_reducesum_nr_threads_lt_blocksize.mlir"
  "integration/reduce/blockwise_reduce/blockwise_reduce_vector_nr_and_scalar_r.mlir"
)

for f in "${FILES[@]}"; do
  echo "$CONTENT" > "$f"
  echo "Fixed: $f"
done
