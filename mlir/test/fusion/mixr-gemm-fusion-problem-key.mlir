// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s

// CHECK: gfx942
// CHECK-SAME: 304
// CHECK-SAME: -t f16 -out_datatype f16 -transA false -transB false -transO false -g 1 -m 100 -n 100 -k 100 -inputFusions=multibroadcast,add,add -outputFusions=relu,add -supportsSplitK false
module {
  func.func @mlir_dot_fused(%arg0: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                            %arg1: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                            %arg2: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                            %arg3: !migraphx.shaped<1x100x1xf16, 100x1x1>)
      -> !migraphx.shaped<1x100x100xf16, 10000x100x1>
      attributes {rock.kernel, rock.arch = "gfx942", rock.num_cu = 304 : i64,
                  rock.input_fusions = ["multibroadcast", "add", "add"],
                  rock.output_fusions = ["relu", "add"]} {
    %0 = migraphx.multibroadcast %arg3 {out_lens = [1, 100, 100], out_dyn_dims = []} : <1x100x1xf16, 100x1x1> -> <1x100x100xf16, 100x1x0>
    %1 = migraphx.add %arg2, %0 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 100x1x0> -> <1x100x100xf16, 10000x100x1>
    %2 = migraphx.add %arg0, %1 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %3 = migraphx.dot %2, %arg1 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %4 = migraphx.relu %3 : <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %5 = migraphx.add %4, %1 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    return %5 : !migraphx.shaped<1x100x100xf16, 10000x100x1>
  }
}
