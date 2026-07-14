// The B operand is fp4 packed along N (rhs_k_pack = false), which the native
// scaled WMMA instruction cannot consume. ScaledBlockedToScaledWMMAF8F6F4 must
// therefore bail and let DecomposeScaledBlocked emulate the dot: both fp4
// operands are upcast with ttg.fp4_to_fp and a plain tt.dot is emitted instead
// of a native tt.dot_scaled.
//
// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s \
// RUN: | rocmlir-gen -ph -fut mlir_dot_fp4 - \
// RUN: | rocmlir-driver -arch gfx1250 -c -o /dev/null \
// RUN:     -mlir-print-ir-after=tritonamdgpu-accelerate-matmul 2>&1 \
// RUN: | FileCheck %s

// CHECK: IR Dump After TritonAMDGPUAccelerateMatmul
// CHECK: ttg.fp4_to_fp
// CHECK: ttg.fp4_to_fp
// CHECK: = tt.dot %
// CHECK-NOT: tt.dot_scaled

func.func @mlir_dot_fp4(%arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>,
                        %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>) -> !migraphx.shaped<1x16x16xf32, 256x16x1> attributes {rock.arch = "gfx1250", rock.kernel = "mixr"} {
  %0 = migraphx.dot %arg0, %arg1 : <1x16x512xf4E2M1FN, 8192x512x1>, <1x512x16xf4E2M1FN, 8192x16x1> -> <1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}
