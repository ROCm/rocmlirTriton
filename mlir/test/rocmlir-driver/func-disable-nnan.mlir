// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `rock.disable-nnan` (`MIGRAPHX_MLIR_DISABLE_NNAN`) withholds only the `nnan`
// flag. Fast math stays on, so migraphx.max is still the non-propagating
// maxnumf; it just does not pick up `nnan`, and a clamp pair therefore cannot
// fold into tt.clampf.

// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN: | rocmlir-opt -rock-allow-fast-math-flags -mlir-print-local-scope \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func @disable_nnan_attr
// CHECK: arith.maxnumf %{{[^,]+}}, %{{[^ ]+}} fastmath<nsz> : tensor<2x3xf32>
// CHECK-NOT: nnan

module {
  func.func @disable_nnan_attr(
      %a: !migraphx.shaped<2x3xf32, 3x1>,
      %b: !migraphx.shaped<2x3xf32, 3x1>) -> !migraphx.shaped<2x3xf32, 3x1>
      attributes {rock.kernel, rock.arch = "gfx90a", "rock.disable-nnan"} {
    %max = migraphx.max %a, %b
        : <2x3xf32, 3x1>, <2x3xf32, 3x1> -> <2x3xf32, 3x1>
    return %max : !migraphx.shaped<2x3xf32, 3x1>
  }
}
