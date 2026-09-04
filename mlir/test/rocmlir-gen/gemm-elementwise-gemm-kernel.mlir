// num_cu is pinned because the attribute list below is checked exactly: when it
// is omitted, rocmlir-gen fills it in from a visible device whose architecture
// matches --arch, so the attribute would appear only on some machines.
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --num_cu 304 --operation gemm_gemm -m 1024 -n 1024 -k 32 -gemmO 32 -t f32 -pv | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK: module attributes {rock.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_gemm_gemm
// CHECK-SAME: (%[[aRaw:.*0]]: tensor<32768xf32>,
// CHECK-SAME: %[[bRaw:.*1]]: tensor<32768xf32>,
// CHECK-SAME: %[[cRaw:.*2]]: tensor<32768xf32>,
// CHECK-SAME: %[[outputRaw:.*3]]: tensor<32768xf32>)
// CHECK-SAME: attributes {rock.arch = "[[$ARCH]]", rock.enable_splitk_for_tuning, rock.kernel, rock.num_cu = 304 : i32}
// CHECK-NEXT: %[[a:.*]] = rock.transform %[[aRaw]] {{.*}} : tensor<32768xf32> to tensor<1x1024x32xf32>
// CHECK-NEXT: %[[b:.*]] = rock.transform %[[bRaw]] {{.*}} : tensor<32768xf32> to tensor<1x32x1024xf32>
// CHECK-NEXT: %[[c:.*]] = rock.transform %[[cRaw]] {{.*}} : tensor<32768xf32> to tensor<1x1024x32xf32>

// CHECK-NEXT: rock.gemm_elementwise_gemm
// CHECK-NEXT: ab = %[[a]] * %[[b]]
// CHECK: out = ab * %[[c]]
// CHECK: %[[flatOutput:.*]] = rock.transform %{{.*}} {{.*}}
// CHECK-NEXT: rock.store %[[flatOutput]] to %[[outputRaw]] by {{.*}}set
// CHECK: return

// CHECK-LABEL: func.func @host_naive_gemm_gemm
// CHECK: %[[abTensor:.*]] = tosa.matmul %[[aTensor:.*]], %[[bTensor:.*]], %{{.*}}, %{{.*}} : ([[aShape:tensor<.*>]], [[bShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]
// CHECK-DAG: %[[resultTensor:.*]] = tosa.matmul %[[abTensor]], %[[cTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : ([[squareShape]], [[cShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[cShape]]
// CHECK: return
