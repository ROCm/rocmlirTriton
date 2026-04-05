// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-gen -fut unpack_dequant_add --arch %arch --clone-harness - | rocmlir-driver -host-pipeline=highlevel -kernel-pipeline=highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut unpack_dequant_add --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base

  // INT4 unpack + dequantize + bias add (elementwise kernel)
  // Input is packed i8 (each byte = 2 int4 values), axis=1 doubles from 128→256
  func.func @unpack_dequant_add(
      %x_packed: !migraphx.shaped<8x128xi8, 128x1>,
      %scale:    !migraphx.shaped<256xf32, 1>,
      %bias:     !migraphx.shaped<256xf32, 1>
  ) -> !migraphx.shaped<8x256xf32, 256x1> attributes {rock.kernel} {
    %unpacked = migraphx.unpack %x_packed {axis = 1 : i64} : <8x128xi8, 128x1> -> <8x256xi8, 256x1>

    %scale_bc = migraphx.multibroadcast %scale {out_dyn_dims = [], out_lens = [8, 256]} : <256xf32, 1> -> <8x256xf32, 0x1>
    %deq = migraphx.dequantizelinear %unpacked, %scale_bc : <8x256xi8, 256x1>, <8x256xf32, 0x1> -> <8x256xf32, 256x1>

    %bias_bc = migraphx.multibroadcast %bias {out_dyn_dims = [], out_lens = [8, 256]} : <256xf32, 1> -> <8x256xf32, 0x1>
    %out = migraphx.add %deq, %bias_bc {} : <8x256xf32, 256x1>, <8x256xf32, 0x1> -> <8x256xf32, 256x1>
    return %out : !migraphx.shaped<8x256xf32, 256x1>
  }
}
