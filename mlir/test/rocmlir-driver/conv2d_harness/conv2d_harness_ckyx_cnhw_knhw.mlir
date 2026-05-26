// RUN: rocmlir-gen --arch %arch -p -fil_layout=gckyx -in_layout=gcnhw -out_layout=gknhw %s | FileCheck %s --check-prefix=HARNESS
// RUN: rocmlir-gen --arch %arch -p -fil_layout=gckyx -in_layout=gcnhw -out_layout=gknhw %s | rocmlir-driver -c | FileCheck %s --check-prefix=LOWERING
// RUN: rocmlir-gen --arch %arch -p -fil_layout=gckyx -in_layout=gcnhw -out_layout=gknhw %s | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

func.func private @rock_conv_gck01_gcn01_gkn01(%arg0: tensor<9216xf32>, %arg1: tensor<1048576xf32>, %arg2: tensor<14745600xf32>) -> tensor<14745600xf32>

// HARNESS: module
// HARNESS: func @rock_conv_gck01_gcn01_gkn01([[FILTER:%.*]]: tensor<9216xf32>, [[INPUT:%.*]]: tensor<1048576xf32>, [[OUTPUT:%.*]]: tensor<14745600xf32>) -> tensor<14745600xf32>
// LOWERING: module
// LOWERING: gpu.binary @rock_kernels

func.func @main() {
  // memref.allocate CPU memory.
  %0 = memref.alloc() : memref<9216xf32>
  %1 = memref.alloc() : memref<1048576xf32>
  %2 = memref.alloc() : memref<14745600xf32>

  // populate initial values.
  %cst = arith.constant 1.0 : f32
  linalg.fill ins(%cst : f32) outs(%0 : memref<9216xf32>)
  linalg.fill ins(%cst : f32) outs(%1 : memref<1048576xf32>)
  linalg.fill ins(%cst : f32) outs(%2 : memref<14745600xf32>)

  // memref.allocate GPU memory.
  %filter = gpu.alloc  () : memref<9216xf32>
  %input = gpu.alloc  () : memref<1048576xf32>
  %output = gpu.alloc  () : memref<14745600xf32>

  // transfer data CPU -> GPU.
  gpu.memcpy  %filter, %0 : memref<9216xf32>, memref<9216xf32>
  gpu.memcpy  %input, %1 : memref<1048576xf32>, memref<1048576xf32>
  gpu.memcpy  %output, %2 : memref<14745600xf32>, memref<14745600xf32>

  // Cast memrefs to tensors for the kernel call.
  %filter_tensor = bufferization.to_tensor %filter {restrict, writable} : memref<9216xf32> to tensor<9216xf32>
  %input_tensor = bufferization.to_tensor %input {restrict, writable} : memref<1048576xf32> to tensor<1048576xf32>
  %output_tensor = bufferization.to_tensor %output {restrict, writable} : memref<14745600xf32> to tensor<14745600xf32>

  // launch kernel.
  %result_tensor = call @rock_conv_gck01_gcn01_gkn01(%filter_tensor, %input_tensor, %output_tensor) : (tensor<9216xf32>, tensor<1048576xf32>, tensor<14745600xf32>) -> tensor<14745600xf32>

  // Convert result tensor back to memref and copy to original output.
  %result_memref = bufferization.to_buffer %result_tensor : tensor<14745600xf32> to memref<14745600xf32>
  memref.copy %result_memref, %output : memref<14745600xf32> to memref<14745600xf32>

  // transfer data GPU -> CPU.
  gpu.memcpy  %2, %output : memref<14745600xf32>, memref<14745600xf32>

  // verify result.
  // TBD. Add more verifying logic.
  %6 = memref.cast %2 : memref<14745600xf32> to memref<*xf32>
  call @printMemrefF32(%6) : (memref<*xf32>) -> ()

  // dellocate GPU memory.
  gpu.dealloc  %filter : memref<9216xf32>
  gpu.dealloc  %input : memref<1048576xf32>
  gpu.dealloc  %output : memref<14745600xf32>

  // memref.deallocate CPU memory.
  memref.dealloc %0 : memref<9216xf32>
  memref.dealloc %1 : memref<1048576xf32>
  memref.dealloc %2 : memref<14745600xf32>

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
// E2E: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [14745600] strides = [1] data =
