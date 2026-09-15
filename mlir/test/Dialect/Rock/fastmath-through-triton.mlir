// Pins what happens to `rock-allow-fast-math-flags`' output once it crosses into
// Triton: for every op that pass can tag and that Triton will legalize, this
// records whether the flags are still there by the time the op is in the LLVM
// dialect.
//
// The input is Rock IR as it stands at the start of `rocmlir-driver -c`, and the
// run below is that same compile minus the `binary` stage, so it stops with the
// kernel in the LLVM dialect. Nothing here writes a fastmath attribute by hand:
// the flags in the expectations can only come from the pass running inside the
// pipeline.

// RUN: rocmlir-driver -arch=gfx942 -kernel-pipeline=gpu,triton %s | FileCheck %s

// The same compile with the pass gated off, which keeps the expectations above
// honest in both directions. It confirms the flags can only have come from the
// pass, and that with no flags to translate the LLVM ops come out bare instead
// of carrying an explicit `fastmath<none>` -- the property that keeps the
// Triton-side translation inert for upstream Triton, whose frontend never sets
// fast-math flags at all.
// RUN: rocmlir-driver -arch=gfx942 --disable-fast-math -kernel-pipeline=gpu,triton %s \
// RUN: | FileCheck %s --check-prefix=NOFM

// gfx1250 is compiled as well because it swaps in a different set of conversions
// for the same source ops: add/sub/mul go through the packed conversion and come
// out as `vector<2x...>`. Without this the packed conversion has no coverage at
// all.
// RUN: rocmlir-driver -arch=gfx1250 -kernel-pipeline=gpu,triton %s \
// RUN: | FileCheck %s --check-prefix=PACKED

#flat_to_gemm = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
  by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>,
      <AddDim{1} ["unit0"] at [0] -> [] at []>]
  bounds = [1, 64, 64] -> [4096]>
#gemm_to_flat = #rock.transform_map<affine_map<(d0) -> (0, d0 floordiv 64, d0 mod 64)>
  by [<Merge{1, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>]
  bounds = [4096] -> [1, 64, 64]>

module attributes {rock.arch = "gfx942"} {
  // Every float `arith` op the pass tags keeps its flags. `divf` is the one that
  // changes codegen the most: without `arcp` the backend cannot use a bare
  // reciprocal and has to emit the fixup sequence. The already-non-propagating
  // `maxnumf` also carries `nnan`, which the propagating `maximumf`/`minimumf`
  // deliberately do not.
  // CHECK-LABEL: llvm.func @arith_f32
  // CHECK-DAG: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : f32
  // CHECK-DAG: llvm.fadd %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fsub %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fneg %{{.*}} {fastmathFlags = #llvm.fastmath<nsz>} : f32
  // CHECK-DAG: llvm.frem %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz>} : f32
  // CHECK-DAG: llvm.intr.maxnum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nnan, nsz>}
  // CHECK-DAG: llvm.intr.maximum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz>}
  // CHECK-DAG: llvm.intr.minimum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz>}
  // CHECK-DAG: llvm.intr.fma(%{{.*}}, %{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz, contract>}
  //
  // NOFM-LABEL: llvm.func @arith_f32
  // NOFM-DAG: llvm.fdiv %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fadd %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fsub %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fmul %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fneg %{{.*}} : f32
  // NOFM-DAG: llvm.frem %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // NOFM-DAG: llvm.intr.maximum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // NOFM-DAG: llvm.intr.minimum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // NOFM-DAG: llvm.intr.fma(%{{.*}}, %{{.*}}, %{{.*}}) : (f32, f32, f32) -> f32
  //
  // The packed conversion applies at f32 too, so it is checked here rather than
  // only on bf16.
  // PACKED-LABEL: llvm.func @arith_f32
  // PACKED-DAG: llvm.fadd %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : vector<2xf32>
  // PACKED-DAG: llvm.fsub %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : vector<2xf32>
  // PACKED-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : vector<2xf32>
  // PACKED-DAG: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : f32
  func.func @arith_f32(%a: tensor<4096xf32>, %b: tensor<4096xf32>,
                       %c: tensor<4096xf32>, %out: tensor<4096xf32>)
      -> tensor<4096xf32> attributes {rock.kernel} {
    %ta = rock.transform %a by #flat_to_gemm : tensor<4096xf32> to tensor<1x64x64xf32>
    %tb = rock.transform %b by #flat_to_gemm : tensor<4096xf32> to tensor<1x64x64xf32>
    %tc = rock.transform %c by #flat_to_gemm : tensor<4096xf32> to tensor<1x64x64xf32>
    %g = rock.gemm %ta * %tb : tensor<1x64x64xf32> * tensor<1x64x64xf32> -> tensor<1x64x64xf32>
    %div = arith.divf %g, %tc : tensor<1x64x64xf32>
    %add = arith.addf %div, %tc : tensor<1x64x64xf32>
    %sub = arith.subf %add, %tc : tensor<1x64x64xf32>
    %mul = arith.mulf %sub, %tc : tensor<1x64x64xf32>
    %neg = arith.negf %mul : tensor<1x64x64xf32>
    %rem = arith.remf %neg, %tc : tensor<1x64x64xf32>
    %maxnum = arith.maxnumf %rem, %tc : tensor<1x64x64xf32>
    %maximum = arith.maximumf %maxnum, %tc : tensor<1x64x64xf32>
    %minimum = arith.minimumf %maximum, %tc : tensor<1x64x64xf32>
    %fma = math.fma %minimum, %tc, %tc : tensor<1x64x64xf32>
    %flat = rock.transform %fma by #gemm_to_flat : tensor<1x64x64xf32> to tensor<4096xf32>
    %s = rock.store %flat to %out by set : tensor<4096xf32> -> tensor<4096xf32> to tensor<4096xf32>
    return %s : tensor<4096xf32>
  }
}
