// The host harness page-locks the buffers it hands to the kernel before they are
// copied to the device. HIP can silently drop small asynchronous host-to-device
// copies made out of pageable memory, which leaves the kernel reading zeros from
// an input that never arrived, with no error reported by any HIP call.

// Every buffer is cast to an unranked memref and registered, and all of that
// happens before the kernel is called.
// RUN: rocmlir-gen --arch %arch --operation gemm -p -ph | FileCheck %s --check-prefix=REGISTER
// REGISTER-LABEL: func.func @main
// REGISTER: %[[BUF:.*]] = memref.cast %{{.*}} to memref<*xf32>
// REGISTER-NEXT: gpu.host_register %[[BUF]] : memref<*xf32>
// REGISTER-COUNT-2: gpu.host_register {{.*}} : memref<*xf32>
// REGISTER: call @rock_gemm{{.*}}_gpu

// Validation runs on the host, so only the three buffers the kernel receives are
// registered and the reference buffers are left alone.
// RUN: rocmlir-gen --arch %arch --operation gemm -g 1 -m 64 -k 64 -n 64 -pv | FileCheck %s --check-prefix=CPUVAL
// CPUVAL-COUNT-3: gpu.host_register {{.*}} : memref<*xf32>
// CPUVAL-NOT: gpu.host_register

// A scaled GEMM registers its two 8-bit scale operands and its f32 output.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 1 -m 1024 -k 768 -n 1024 -t f4E2M1FN -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU -out_dtype f32 --scaledGemm -pv | FileCheck %s --check-prefix=SCALED
// SCALED-COUNT-2: gpu.host_register {{.*}} : memref<*xf8E8M0FNU>
// SCALED: gpu.host_register {{.*}} : memref<*xf32>
// SCALED: call @rock_gemm_gpu

// Its f4E2M1FN operands are skipped, because a sub-byte memref cannot be cast to
// an unranked memref. This prefix carries only a CHECK-NOT, so that it applies to
// the whole harness rather than to whatever follows an earlier match.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 1 -m 1024 -k 768 -n 1024 -t f4E2M1FN -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU -out_dtype f32 --scaledGemm -pv | FileCheck %s --check-prefix=NO-SUBBYTE
// NO-SUBBYTE-NOT: memref<*xf4E2M1FN>
