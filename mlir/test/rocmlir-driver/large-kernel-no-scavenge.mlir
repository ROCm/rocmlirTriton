// UNSUPPORTED: true
// TODO(rocmlirTriton): external/triton/llvm-project/mlir/lib/IR/Operation.cpp:509: void llvm::ilist_traits<mlir::Operation>::removeNodeFromList(Operation *): Assertion `op->block && "not already in an operation block!"' failed.

// RUN: rocmlir-gen --operation conv_bwd_data -t f16 --arch %arch --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 192 --in_h 64 --in_w 64 --out_channels 384 --fil_h 4 --fil_w 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --groupsize 1 | rocmlir-driver --kernel-pipeline=full | FileCheck %s

// CHECK: rock.grid_size{{.*}} = {{.*}} : i32, triton.hsaco

