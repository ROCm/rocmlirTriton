// The kernel path may assume no NaN operands; the CPU reference may not. It
// exists to say what the kernel ought to have computed, so if it were lowered
// under the same assumption the comparison would be circular -- both sides would
// be free to disagree with IEEE in the same direction and still match.
//
// rocmlir-driver runs the migraphx and highlevel phases once per side, and only
// the kernel run is given `-disable-fast-math`'s value; the host run is pinned to
// the IEEE behaviour. These tests pin that split down at each of the three
// places the relaxation could otherwise reach the reference.

// Leak 1: `migraphx-to-tosa` picks the NaN mode. IGNORE is a kernel-only choice.
// RUN: rocmlir-gen --clone-harness -arch %arch -fut host_ieee %s \
// RUN: | rocmlir-driver -arch=%arch -kernel-pipeline=migraphx -host-pipeline=migraphx -o - \
// RUN: | FileCheck %s --check-prefix=TOSA

// `nan_mode` prints only when it is not the default PROPAGATE, so the bare ops
// here are the IEEE ones.
// TOSA-LABEL: func.func @host_ieee_cpu_host
// TOSA: tosa.maximum
// TOSA-NOT: nan_mode
// TOSA: tosa.minimum
// TOSA-NOT: nan_mode

// TOSA-LABEL: func.func @host_ieee
// TOSA-SAME: rock.kernel
// TOSA: tosa.maximum {{.*}}nan_mode = IGNORE
// TOSA: tosa.minimum {{.*}}nan_mode = IGNORE

// Leak 2: `rock-tosa-to-elementwise` picks between the propagating min/max pair
// and the num pair. The host never reaches it (it goes through upstream
// tosa-to-linalg), which is what leaves it on the propagating pair.
// RUN: rocmlir-gen --clone-harness -arch %arch -fut host_ieee %s \
// RUN: | rocmlir-driver -arch=%arch -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -o - \
// RUN: | FileCheck %s --check-prefix=ROCK

// ROCK-LABEL: func.func @host_ieee_cpu_host
// ROCK: arith.maximumf
// ROCK: arith.minimumf

// ROCK-LABEL: func.func @host_ieee
// ROCK-SAME: rock.kernel
// ROCK: arith.maxnumf
// ROCK: arith.minnumf

// Leak 3: `rock-allow-fast-math-flags` runs while the host funcs are still in
// the module, so only its `rock.kernel` gate keeps their flags off. By the end
// of the kernel pipeline the reference has been serialized into the
// `rock.host_functions` string, where it must still read as plain IEEE: the
// operand-only regexes below cannot match if a `fastmath<...>` has appeared
// between the last operand and the type.
// RUN: rocmlir-gen --clone-harness -arch %arch -fut host_ieee %s \
// RUN: | rocmlir-driver -arch=%arch -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | rocmlir-driver -arch=%arch -kernel-pipeline=gpu -o - \
// RUN: | FileCheck %s --check-prefix=FLAGS

// FLAGS: rock.host_functions
// FLAGS-SAME: arith.divf %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : f32
// FLAGS-SAME: arith.maximumf %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : f32
// FLAGS-SAME: arith.minimumf %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : f32

// The kernel meanwhile did take the relaxation: `nnan` on the clamp pair is
// what let it fold to a single tt.clampf that need not propagate NaNs.
// FLAGS: tt.func @host_ieee
// FLAGS: tt.clampf
// FLAGS-SAME: propagateNan = none

module {
  func.func @host_ieee(%a: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                       %b: !migraphx.shaped<1x256x256xf32, 65536x256x1>)
      -> !migraphx.shaped<1x256x256xf32, 65536x256x1>
      attributes {rock.kernel} {
    %lo = migraphx.literal (dense<0.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %hi = migraphx.literal (dense<6.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %blo = migraphx.multibroadcast %lo {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf32, 0> -> <1x256x256xf32, 0x0x0>
    %bhi = migraphx.multibroadcast %hi {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf32, 0> -> <1x256x256xf32, 0x0x0>
    %d = migraphx.dot %a, %b : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    // Divided by an input rather than a constant so the divf survives instead of
    // folding to a multiply: it is the op a leak would be loudest on, taking
    // `arcp` and `afn` and turning the reference's division into a reciprocal.
    %q = migraphx.div %d, %a : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %clipped = migraphx.clip %q, %blo, %bhi : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 0x0x0>, <1x256x256xf32, 0x0x0> -> <1x256x256xf32, 65536x256x1>
    return %clipped : !migraphx.shaped<1x256x256xf32, 65536x256x1>
  }
}
