// convertDistributedOpEncoding converts the relayouted load's result back, so
// the pass on its own leaves a no-op conversion pair where the old conversion
// was. Canonicalization is what removes it, and the pass's contract is about
// what survives that, so it is part of the run.
// RUN: rocmlir-opt -split-input-file --rock-unify-dot-operand-loads --canonicalize %s | FileCheck %s
// The LDS consequence is the point of the pass, so it is checked directly.
// remove-layout-conversions is what folds away the conversions the pass leaves
// behind, and allocate-shared-memory is what reports the peak.
// RUN: rocmlir-opt -split-input-file --rock-unify-dot-operand-loads --canonicalize --tritongpu-remove-layout-conversions --allocate-shared-memory %s | FileCheck %s --check-prefix=LDS

// An input fusion whose two loads coalesce differently. The fused input is
// reissued in the layout of the larger load, so the conversion that used to
// join them is gone and only its pointer tensor is converted, which is index
// arithmetic that folds back into the splat and make_range it came from.
//
// The LDS check is the invariant the whole budget contract rests on: the fused
// kernel allocates what the unfused one below does (4096, the algorithmic
// blocked -> dot_op staging), rather than the 8192 it would allocate if the
// fusion's own conversion survived.
// CHECK-LABEL: @fused_input_unified
//      CHECK:   %[[A:.+]] = tt.load %{{.*}} {rock.load_tensor_bytes = 32768 : i64} : tensor<64x64x!tt.ptr<f16>, #[[BIG:.+]]>
//      CHECK:   %[[FPTR:.+]] = ttg.convert_layout %{{.*}} : tensor<64x64x!tt.ptr<f16>, #{{.+}}> -> tensor<64x64x!tt.ptr<f16>, #[[BIG]]>
//      CHECK:   %[[F:.+]] = tt.load %[[FPTR]] {rock.load_tensor_bytes = 8192 : i64} : tensor<64x64x!tt.ptr<f16>, #[[BIG]]>
// The add now consumes both loads directly: the conversion that used to join
// them, and the LDS it needed, are gone.
//      CHECK:   arith.addf %[[A]], %[[F]] : tensor<64x64xf16, #[[BIG]]>
// LDS: ttg.shared = 4096
// LDS-LABEL: @fused_input_unified
#blockedBig = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedSmall = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fused_input_unified(%abase: !tt.ptr<f16>, %fbase: !tt.ptr<f16>, %bop: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) -> tensor<64x64xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    %r0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blockedBig}>>
    %r0e = tt.expand_dims %r0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blockedBig}>> -> tensor<1x64xi32, #blockedBig>
    %r0b = tt.broadcast %r0e : tensor<1x64xi32, #blockedBig> -> tensor<64x64xi32, #blockedBig>
    %sa = tt.splat %abase : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blockedBig>
    %pa = tt.addptr %sa, %r0b : tensor<64x64x!tt.ptr<f16>, #blockedBig>, tensor<64x64xi32, #blockedBig>
    %a = tt.load %pa {rock.load_tensor_bytes = 32768 : i64} : tensor<64x64x!tt.ptr<f16>, #blockedBig>

    %r1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blockedSmall}>>
    %r1e = tt.expand_dims %r1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blockedSmall}>> -> tensor<1x64xi32, #blockedSmall>
    %r1b = tt.broadcast %r1e : tensor<1x64xi32, #blockedSmall> -> tensor<64x64xi32, #blockedSmall>
    %sf = tt.splat %fbase : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blockedSmall>
    %pf = tt.addptr %sf, %r1b : tensor<64x64x!tt.ptr<f16>, #blockedSmall>, tensor<64x64xi32, #blockedSmall>
    %f = tt.load %pf {rock.load_tensor_bytes = 8192 : i64} : tensor<64x64x!tt.ptr<f16>, #blockedSmall>

    %fc = ttg.convert_layout %f : tensor<64x64xf16, #blockedSmall> -> tensor<64x64xf16, #blockedBig>
    %sum = arith.addf %a, %fc : tensor<64x64xf16, #blockedBig>
    %aop = ttg.convert_layout %sum : tensor<64x64xf16, #blockedBig> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %d = tt.dot %aop, %bop, %acc : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----

// The same kernel without the fusion, which is what the cached perf config was
// tuned against and so what the LDS figure above has to match.
// LDS: ttg.shared = 4096
// LDS-LABEL: @unfused_reference
#blockedBig = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @unfused_reference(%abase: !tt.ptr<f16>, %bop: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) -> tensor<64x64xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %r0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blockedBig}>>
    %r0e = tt.expand_dims %r0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blockedBig}>> -> tensor<1x64xi32, #blockedBig>
    %r0b = tt.broadcast %r0e : tensor<1x64xi32, #blockedBig> -> tensor<64x64xi32, #blockedBig>
    %sa = tt.splat %abase : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blockedBig>
    %pa = tt.addptr %sa, %r0b : tensor<64x64x!tt.ptr<f16>, #blockedBig>, tensor<64x64xi32, #blockedBig>
    %a = tt.load %pa {rock.load_tensor_bytes = 32768 : i64} : tensor<64x64x!tt.ptr<f16>, #blockedBig>
    %aop = ttg.convert_layout %a : tensor<64x64xf16, #blockedBig> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %d = tt.dot %aop, %bop, %acc : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----

// Both loads already agree, the common no-transpose case. Nothing is rewritten,
// so no pointer conversion appears.
// CHECK-LABEL: @layouts_already_agree
//  CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<64x64x!tt.ptr<f16>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @layouts_already_agree(%abase: !tt.ptr<f16>, %fbase: !tt.ptr<f16>, %bop: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) -> tensor<64x64xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %sa = tt.splat %abase : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %a = tt.load %sa {rock.load_tensor_bytes = 32768 : i64} : tensor<64x64x!tt.ptr<f16>, #blocked>
    %sf = tt.splat %fbase : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %f = tt.load %sf {rock.load_tensor_bytes = 8192 : i64} : tensor<64x64x!tt.ptr<f16>, #blocked>
    %sum = arith.addf %a, %f : tensor<64x64xf16, #blocked>
    %aop = ttg.convert_layout %sum : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %d = tt.dot %aop, %bop, %acc : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----

// Neither load is tagged, so this is not a chain rock built and there is no
// basis for calling one of them the leader. Left alone rather than guessed at.
// CHECK-LABEL: @untagged_loads_left_alone
//      CHECK:   ttg.convert_layout %{{.*}} : tensor<64x64xf16, #{{.+}}> -> tensor<64x64xf16, #{{.+}}>
#blockedBig = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedSmall = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @untagged_loads_left_alone(%abase: !tt.ptr<f16>, %fbase: !tt.ptr<f16>, %bop: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) -> tensor<64x64xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %sa = tt.splat %abase : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blockedBig>
    %a = tt.load %sa : tensor<64x64x!tt.ptr<f16>, #blockedBig>
    %sf = tt.splat %fbase : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blockedSmall>
    %f = tt.load %sf : tensor<64x64x!tt.ptr<f16>, #blockedSmall>
    %fc = ttg.convert_layout %f : tensor<64x64xf16, #blockedSmall> -> tensor<64x64xf16, #blockedBig>
    %sum = arith.addf %a, %fc : tensor<64x64xf16, #blockedBig>
    %aop = ttg.convert_layout %sum : tensor<64x64xf16, #blockedBig> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %d = tt.dot %aop, %bop, %acc : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}
