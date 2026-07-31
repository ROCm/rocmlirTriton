// Verify that the two ReLUs in this input-fused convolution remain
// arith.maximumf operations through Rock-to-TTIR. This guards the GPU
// pipeline's disabled floating-point min/max expansion.

// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_test %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-driver -c -arch %arch -mlir-print-ir-after=rock-to-ttir -o /dev/null 2>&1 | FileCheck %s --check-prefix=TTIR
// TTIR-LABEL: IR Dump After {{.*}}rock-to-ttir
// Both ReLUs must survive as maximum operations.
// TTIR-COUNT-2: arith.maximumf

module {
  func.func @mlir_test(%arg0: !migraphx.shaped<544x1x1xf16, 1x1x1>, %arg1: !migraphx.shaped<32x544x7x7xf16, 28224x49x7x1>, %arg2: !migraphx.shaped<544xf16, 1>, %arg3: !migraphx.shaped<128x544x1x1xf16, 544x1x1x1>, %arg4: !migraphx.shaped<128xf16, 1>) -> !migraphx.shaped<32x128x7x7xf16, 6272x49x7x1> attributes {rock.kernel} {
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [32, 544, 7, 7]} : <544x1x1xf16, 1x1x1> -> <32x544x7x7xf16, 0x1x0x0>
    %1 = migraphx.broadcast %arg2 {axis = 1 : i64, out_lens = [32, 544, 7, 7]} : <544xf16, 1> -> <32x544x7x7xf16, 0x1x0x0>
    %2 = migraphx.mul %0, %arg1 : <32x544x7x7xf16, 0x1x0x0>, <32x544x7x7xf16, 28224x49x7x1> -> <32x544x7x7xf16, 26656x49x7x1>
    %3 = migraphx.add %2, %1 : <32x544x7x7xf16, 26656x49x7x1>, <32x544x7x7xf16, 0x1x0x0> -> <32x544x7x7xf16, 26656x49x7x1>
    %4 = migraphx.relu %3 : <32x544x7x7xf16, 26656x49x7x1> -> <32x544x7x7xf16, 26656x49x7x1>
    %5 = migraphx.convolution %4, %arg3 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <32x544x7x7xf16, 26656x49x7x1>, <128x544x1x1xf16, 544x1x1x1> -> <32x128x7x7xf16, 6272x49x7x1>
    %6 = migraphx.broadcast %arg4 {axis = 1 : i64, out_lens = [32, 128, 7, 7]} : <128xf16, 1> -> <32x128x7x7xf16, 0x1x0x0>
    %7 = migraphx.add %5, %6 : <32x128x7x7xf16, 6272x49x7x1>, <32x128x7x7xf16, 0x1x0x0> -> <32x128x7x7xf16, 6272x49x7x1>
    %8 = migraphx.relu %7 : <32x128x7x7xf16, 6272x49x7x1> -> <32x128x7x7xf16, 6272x49x7x1>
    return %8 : !migraphx.shaped<32x128x7x7xf16, 6272x49x7x1>
  }
}
