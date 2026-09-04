// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f16 -pv | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK: module attributes {rock.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: tensor<32768xf16>,
// CHECK-SAME: %[[keysRaw:.*1]]: tensor<32768xf16>,
// CHECK-SAME: %[[valuesRaw:.*2]]: tensor<32768xf16>,
// CHECK-SAME: %[[scaleRaw:.*3]]: tensor<1048576xf16>,
// CHECK-SAME: %[[outputRaw:.*4]]: tensor<32768xf16>)
// CHECK-SAME: attributes {rock.arch = "[[$ARCH]]", rock.kernel{{.*}}}
// CHECK-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : tensor<32768xf16> to tensor<1x1024x32xf16>
// CHECK-NEXT: %[[keys:.*]] = rock.transform %[[keysRaw]] {{.*}} : tensor<32768xf16> to tensor<1x32x1024xf16>
// CHECK-NEXT: %[[values:.*]] = rock.transform %[[valuesRaw]] {{.*}} : tensor<32768xf16> to tensor<1x1024x32xf16>
// CHECK-NEXT: %[[scale:.*]] = rock.transform %[[scaleRaw]] {{.*}} : tensor<1048576xf16> to tensor<1x1024x1024xf16>

// CHECK-NEXT: %[[output:.*]] = rock.attention
// CHECK-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK-NEXT: qk = elementwise otherIns(%[[scale]]
// CHECK: softmax(qk) * %[[values]]
// CHECK: %[[flatOutput:.*]] = rock.transform %{{.*}} {{.*}}
// CHECK-NEXT: rock.store %[[flatOutput]] to %[[outputRaw]] by {{.*}}set
// CHECK: return

// CHECK-LABEL: func.func @host_naive_attention
// CHECK: %[[qkTensor:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf16>, tensor<1xf16>) -> [[squareShape:tensor<.*>]]
// CHECK-DAG: %[[sqkTensor:.*]] = tosa.mul %[[qkTensor]], %[[scaleTensor:.*]], %{{.*}} : ([[squareShape]], [[squareShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK-DAG: %[[sqkTensorCast:.*]] = tosa.cast %[[sqkTensor]] : ([[squareShape]]) -> [[squareShapeF32:tensor<.*>]]
// CHECK-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[sqkTensorCast]] {{.*}} : ([[squareShapeF32]]) -> [[reducedShape:tensor<.*>]]
// CHECK-DAG: %[[normilizedSqkTensor:.*]] = tosa.sub %[[sqkTensorCast]], %[[sqkMaxs]] : ([[squareShapeF32]], [[reducedShape]]) -> [[squareShapeF32]]
// CHECK-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedSqkTensor]] : ([[squareShapeF32]]) -> [[squareShapeF32]]
// CHECK-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShapeF32]]) -> [[reducedShape]]
// CHECK-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShapeF32]], [[reducedShape]], tensor<1xi8>) -> [[squareShapeF32]] 
// CHECK-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShapeF32]]) -> [[squareShape]]
// CHECK-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : ([[squareShape]], [[valuesShape:tensor<.*>]], tensor<1xf16>, tensor<1xf16>) -> [[squareShape:tensor<.*>]]
// CHECK: return

// `--softmax_dtype` controls the type used for the softmax intermediate
// inside `rock.attention`. The default for an f16 input kernel is f32
// (wider than the operand type for numerical stability); requesting f16
// collapses the intermediate to the operand type to save footprint.
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 | FileCheck %s --check-prefix=SOFTMAX_DEFAULT_F16
// SOFTMAX_DEFAULT_F16: rock.attention
// SOFTMAX_DEFAULT_F16: softmaxType = f32
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 --softmax_dtype f16 | FileCheck %s --check-prefix=SOFTMAX_F16
// SOFTMAX_F16: rock.attention
// SOFTMAX_F16: softmaxType = f16
