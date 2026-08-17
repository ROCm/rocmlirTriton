// Arch is fixed here because not all architectures have atomic_add
// RUN: rocmlir-gen --arch gfx908 --operation gemm -p --store-method atomic_add | FileCheck %s --check-prefix=ATOMIC_ADD
// ATOMIC_ADD: rock.gemm
// ATOMIC_ADD: rock.store {{.*}} by atomic_add
// RUN: rocmlir-gen --emit-tuning-key -p --arch gfx1101 | FileCheck %s --check-prefix=CONV
// CONV: amdgcn-amd-amdhsa:gfx1101   {{.*}}     conv -F 1 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1
// RUN: rocmlir-gen --emit-tuning-key -p -t fp8_fp8 --arch gfx1101 | FileCheck %s --check-prefix=CONVFP8
// CONVFP8: amdgcn-amd-amdhsa:gfx1101   {{.*}}     convfp8_fp8 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1
// RUN: rocmlir-gen --emit-tuning-key -p -t fp8_fp8 --arch gfx1201 | FileCheck %s --check-prefix=CONVOCPFP8
// CONVOCPFP8: amdgcn-amd-amdhsa:gfx1201   {{.*}}     convfp8_fp8 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1
// RUN: rocmlir-gen --arch gfx908 --operation gemm -p --emit-tuning-key | FileCheck %s --check-prefix=GEMM
// GEMM: amdgcn-amd-amdhsa:gfx908   {{.*}}     -t f32 -out_datatype f32 -transA false -transB false -transO false -g 1 -m 1024 -n 512 -k 769 -supportsSplitK true
// rocmlirTriton's atomic-add CAS fallback also makes i32 outputs split-K capable.
// RUN: rocmlir-gen --arch gfx908 --operation gemm -t i8 -p --emit-tuning-key | FileCheck %s --check-prefix=GEMM_I8
// GEMM_I8: amdgcn-amd-amdhsa:gfx908   {{.*}}     -t i8 -out_datatype i32 -transA false -transB false -transO false -g 1 -m 1024 -n 512 -k 769 -supportsSplitK true{{$}}
// RUN: rocmlir-gen --emit-tuning-key -p -t fp8_fp8 --arch gfx950 | FileCheck %s --check-prefix=CONVOCPFP8_GFX950
// CONVOCPFP8_GFX950: amdgcn-amd-amdhsa:gfx950   {{.*}}     convfp8_fp8 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -p --num_cu 40 --num_chiplets 20 | FileCheck %s --check-prefix=NUM_CHIPLETS
// NUM_CHIPLETS: rock.num_chiplets = 20 : i64, rock.num_cu = 40 : i64

// `--emit-tuning-key` for backward-data uses the same conv key payload
// as forward, but with `-F 2` instead of `-F 1`.
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_data -p -t f32 --emit-tuning-key | FileCheck %s --check-prefix=BWD_DATA_KEY
// BWD_DATA_KEY: amdgcn-amd-amdhsa:gfx942   {{.*}}     conv -F 2 -f GNC01 -I NGC01 -O NGC01 -n 128 -c 8 -H 32 -W 32 -k 128 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1

// `-pi` (`--print-inputs`) prints every input tensor of the host harness
// (all kernel args except the output). For a 3-arg GEMM that is A and B,
// emitted as two `printMemrefF32` calls. `-pr` and `-pvr` are already
// covered by populate_host_print*.mlir and the fusion E2E tests.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph -pi | FileCheck %s --check-prefix=PRINT_INPUTS
// PRINT_INPUTS-LABEL: func.func @main()
// PRINT_INPUTS-COUNT-2: call @printMemrefF32
// PRINT_INPUTS-NOT: call @printMemrefF32

// `--print-verify-results=<level>` controls the verbosity of the verification
// output (off=0, summary=1, failure=2, always=3). Summary is the default.
// The level is forwarded as the trailing `i8` constant in the call.
//
// Default comparator (allclose): emits `mcpuVerifyFloatAllclose` with
// signature (memref<?xf32>, memref<?xf32>, f32, f32, i8, i1).
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -pv | FileCheck %s --check-prefix=VERIFY_SUMMARY
// VERIFY_SUMMARY: %[[lvl:.*]] = arith.constant 1 : i8
// VERIFY_SUMMARY: call @mcpuVerifyFloatAllclose({{.*}}, %[[lvl]], %{{.*}}) : (memref<?xf32>, memref<?xf32>, f32, f32, i8, i1) -> ()
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -pv --print-verify-results=always | FileCheck %s --check-prefix=VERIFY_ALWAYS
// VERIFY_ALWAYS: %[[lvl:.*]] = arith.constant 3 : i8
// VERIFY_ALWAYS: call @mcpuVerifyFloatAllclose({{.*}}, %[[lvl]], %{{.*}}) : (memref<?xf32>, memref<?xf32>, f32, f32, i8, i1) -> ()
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -pv --print-verify-results=off | FileCheck %s --check-prefix=VERIFY_OFF
// VERIFY_OFF: %[[lvl:.*]] = arith.constant 0 : i8
// VERIFY_OFF: call @mcpuVerifyFloatAllclose({{.*}}, %[[lvl]], %{{.*}}) : (memref<?xf32>, memref<?xf32>, f32, f32, i8, i1) -> ()
//
// Legacy comparator: emits `mcpuVerifyFloat` with
// signature (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1).
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -pv --comparator=legacy | FileCheck %s --check-prefix=VERIFY_LEGACY
// VERIFY_LEGACY: call @mcpuVerifyFloat({{.*}}) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1) -> ()
// Same flag for integer kernels through `mcpuVerifyInt32Int32`.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t i8 -out_datatype i32 -g 1 -m 32 -n 32 -k 32 -pv --print-verify-results=failure | FileCheck %s --check-prefix=VERIFY_INT
// VERIFY_INT: %[[lvl:.*]] = arith.constant 2 : i8
// VERIFY_INT: call @mcpuVerifyInt32Int32({{.*}}, %[[lvl]]) : (memref<?xi32>, memref<?xi32>, i8) -> ()

// `--cpu-timers` injects load-to-main / init / GPU / CPU timer hooks.
// With `-ph` only (no CPU reference), the CPU timer pair is absent.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph --cpu-timers | FileCheck %s --check-prefix=TIMERS_HOST_ONLY
// TIMERS_HOST_ONLY-LABEL: func.func @main()
// TIMERS_HOST_ONLY: call @programStart()
// TIMERS_HOST_ONLY: call @initTimerStart()
// TIMERS_HOST_ONLY: call @initTimerStop()
// TIMERS_HOST_ONLY: call @gpuTimerStart()
// TIMERS_HOST_ONLY: call @gpuTimerStop()
// TIMERS_HOST_ONLY-NOT: call @cpuTimerStart()

// Combining `--cpu-timers` with `-pv` adds the CPU reference timer pair
// around the host gemm call.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -pv --cpu-timers | FileCheck %s --check-prefix=TIMERS_WITH_PV
// TIMERS_WITH_PV-LABEL: func.func @main()
// TIMERS_WITH_PV: call @programStart()
// TIMERS_WITH_PV: call @initTimerStart()
// TIMERS_WITH_PV: call @initTimerStop()
// TIMERS_WITH_PV: call @gpuTimerStart()
// TIMERS_WITH_PV: call @gpuTimerStop()
// TIMERS_WITH_PV: call @cpuTimerStart()
// TIMERS_WITH_PV: call @cpuTimerStop()

// `--device <N>` (and its `-dev` alias) registers a global constructor that
// calls `gpu.set_default_device` with the requested device index. The
// constructor is only emitted when the flag is actually passed.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph | FileCheck %s --check-prefix=NO_DEVICE
// NO_DEVICE-NOT: llvm.func @setDeviceCtor
// NO_DEVICE-NOT: gpu.set_default_device
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph --device 1 | FileCheck %s --check-prefix=DEVICE_1
// DEVICE_1: llvm.func @setDeviceCtor()
// DEVICE_1: %[[idx:.*]] = arith.constant 1 : i32
// DEVICE_1: gpu.set_default_device %[[idx]]
// DEVICE_1: llvm.mlir.global_ctors ctors = [@setDeviceCtor], priorities = [122 : i32], data = [#llvm.zero]
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph -dev 2 | FileCheck %s --check-prefix=DEVICE_2
// DEVICE_2: llvm.func @setDeviceCtor()
// DEVICE_2: %[[idx:.*]] = arith.constant 2 : i32
// DEVICE_2: gpu.set_default_device %[[idx]]
