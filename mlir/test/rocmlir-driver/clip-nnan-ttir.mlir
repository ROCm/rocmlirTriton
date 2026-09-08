// The middle of the story that clip-nnan-isa.mlir only measures the end of: a
// `migraphx.clip` arrives in TTIR as one `tt.clampf`. Every stage has to agree
// for that to happen -- MIGraphXToTosa must set `nan_mode = IGNORE`,
// RockTosaToElementwise must pick minnumf/maxnumf over the propagating pair,
// RockAllowFastMathFlags must mark them `nnan`, and only then does RockToTTIR
// fold them. `-disable-fast-math` turns each of those off, so the two runs
// below also serve as a check that the flag reaches every one of those stages.
//
// `-kernel-pipeline=gpu` stops right after RockToTTIR; running on to `triton`
// would lower tt.clampf away again.

// RUN: rocmlir-gen --clone-harness -arch %arch -fut clip_nnan %s \
// RUN: | rocmlir-driver -arch=%arch -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | rocmlir-driver -arch=%arch -kernel-pipeline=gpu -o - \
// RUN: | FileCheck %s

// The kernel keeps no min/max of its own. The CPU reference clone keeps a
// propagating pair -- the driver pins the host run to IEEE, see
// host-keeps-ieee-nan.mlir -- but either way none of it lands in the tt.func.
// CHECK-LABEL: tt.func @clip_nnan
// CHECK-NOT: arith.minnumf
// CHECK-NOT: arith.maxnumf
// CHECK: tt.clampf
// CHECK-SAME: propagateNan = none

// Under -disable-fast-math the clip stays on the NaN-propagating pair, so there
// is nothing for the fold to match. The flag has to be given to both driver
// invocations: the first covers migraphx-to-tosa's NaN mode and
// rock-tosa-to-elementwise, the second gates rock-allow-fast-math-flags and
// rock-to-ttir.
// RUN: rocmlir-gen --clone-harness -arch %arch -fut clip_nnan %s \
// RUN: | rocmlir-driver -disable-fast-math -arch=%arch -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | rocmlir-driver -disable-fast-math -arch=%arch -kernel-pipeline=gpu -o - \
// RUN: | FileCheck %s --check-prefix=IEEE

// IEEE-LABEL: tt.func @clip_nnan
// IEEE-NOT: tt.clampf
// IEEE: arith.maximumf
// IEEE: arith.minimumf

module {
  func.func @clip_nnan(%a: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                       %b: !migraphx.shaped<1x256x256xf32, 65536x256x1>)
      -> !migraphx.shaped<1x256x256xf32, 65536x256x1>
      attributes {rock.kernel} {
    %lo = migraphx.literal (dense<0.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %hi = migraphx.literal (dense<6.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %blo = migraphx.multibroadcast %lo {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf32, 0> -> <1x256x256xf32, 0x0x0>
    %bhi = migraphx.multibroadcast %hi {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf32, 0> -> <1x256x256xf32, 0x0x0>
    %d = migraphx.dot %a, %b : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %clipped = migraphx.clip %d, %blo, %bhi : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 0x0x0>, <1x256x256xf32, 0x0x0> -> <1x256x256xf32, 65536x256x1>
    return %clipped : !migraphx.shaped<1x256x256xf32, 65536x256x1>
  }
}
