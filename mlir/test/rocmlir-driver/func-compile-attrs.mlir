// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// MIGraphX requests `-disable-fast-math` by stamping `rock.disable-fast-math`
// on the kernel function (`MIGRAPHX_MLIR_DISABLE_FAST_MATH`). The driver has
// to carry that into migraphx-to-tosa: that is what flips `tosa.maximum` from
// IGNORE to the default PROPAGATE, which then prints with no `nan_mode`.

// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=migraphx %s -o - 2>%t.pipelines \
// RUN: | FileCheck %s --check-prefix=FASTOFF
// RUN: FileCheck %s --check-prefix=PIPE --input-file=%t.pipelines

// `nan_mode` prints only when it is not the default PROPAGATE.
// FASTOFF-LABEL: func.func @disable_fast_math_attr
// FASTOFF-NOT: nan_mode
// PIPE: migraphx-to-tosa{disable-fast-math=true}

module {
  func.func @disable_fast_math_attr(
      %a: !migraphx.shaped<2x3xf32, 3x1>,
      %b: !migraphx.shaped<2x3xf32, 3x1>) -> !migraphx.shaped<2x3xf32, 3x1>
      attributes {rock.kernel, rock.arch = "gfx90a", "rock.disable-fast-math"} {
    %max = migraphx.max %a, %b
        : <2x3xf32, 3x1>, <2x3xf32, 3x1> -> <2x3xf32, 3x1>
    return %max : !migraphx.shaped<2x3xf32, 3x1>
  }
}
