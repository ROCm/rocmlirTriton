// `ninf` is the one fast-math flag the tanh approximation in
// rock-legalize-math-for-triton cannot survive. It reaches -1 by letting the
// exponential overflow to +inf and taking the reciprocal of that, which is
// exactly +0, so a flag that lets the backend assume no operand is ever
// infinite would turn that whole end of the curve into a NaN.
//
// The approximation asks for no flags of its own; rock-allow-fast-math-flags
// attaches them afterwards, which is where `ninf` would come from if it came
// from anywhere. So this runs the real pipeline over a real kernel and dumps
// the IR straight after that pass, and the --implicit-check-not rejects the
// flag on every op in the dump rather than only on the tanh.
//
// This is the static half. The -4..4 RUN of mixr-conv-tanh-accuracy.e2e.mlir is
// the other one: it drives both ends into saturation on hardware and would
// print NaN rather than +/-1 if the flag ever appeared.

// RUN: rocmlir-gen --clone-harness -arch gfx942 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx942 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | rocmlir-driver -arch=gfx942 -c -mlir-print-ir-after=rock-allow-fast-math-flags -o /dev/null 2>&1 \
// RUN: | FileCheck %s --implicit-check-not=ninf

// The flags the approximation does end up with. Checking those is what keeps
// the test honest: without them, a change that stopped the pass attaching any
// flags at all would satisfy the implicit-check-not on its own. `arcp` on the
// divide is the one that matters, since it is what makes it a v_rcp_f32.
// CHECK: math.exp2 %{{.*}} fastmath<nsz,contract,afn>
// CHECK: arith.divf %{{.*}} fastmath<nsz,arcp,afn>

module {
  func.func @mlir_convolution_tanh(%arg0: !migraphx.shaped<1x8x4x4xf16, 128x16x4x1>, %arg1: !migraphx.shaped<8x8x3x3xf16, 72x9x3x1>) -> !migraphx.shaped<1x8x4x4xf16, 128x16x4x1> attributes {rock.kernel} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf16, 128x16x4x1>, <8x8x3x3xf16, 72x9x3x1> -> <1x8x4x4xf16, 128x16x4x1>
    %1 = migraphx.tanh %0 : <1x8x4x4xf16, 128x16x4x1> -> <1x8x4x4xf16, 128x16x4x1>
    return %1 : !migraphx.shaped<1x8x4x4xf16, 128x16x4x1>
  }
}
