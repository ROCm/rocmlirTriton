// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f32 -pv | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_SCALE

// CHECK_SCALE: module attributes {rock.arch = "[[$ARCH:.*]]"}

// CHECK_SCALE-LABEL: func.func @rock_attention
// CHECK_SCALE-SAME: (%[[queriesRaw:.*0]]: tensor<32768xf32>,
// CHECK_SCALE-SAME: %[[keysRaw:.*1]]: tensor<32768xf32>,
// CHECK_SCALE-SAME: %[[valuesRaw:.*2]]: tensor<32768xf32>,
// CHECK_SCALE-SAME: %[[scaleRaw:.*3]]: tensor<1048576xf32>,
// CHECK_SCALE-SAME: %[[outputRaw:.*4]]: tensor<32768xf32>)
// CHECK_SCALE-SAME: attributes {rock.arch = "[[$ARCH]]", rock.kernel}
// CHECK_SCALE-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : tensor<32768xf32> to tensor<1x1024x32xf32>
// CHECK_SCALE-NEXT: %[[keys:.*]] = rock.transform %[[keysRaw]] {{.*}} : tensor<32768xf32> to tensor<1x32x1024xf32>
// CHECK_SCALE-NEXT: %[[values:.*]] = rock.transform %[[valuesRaw]] {{.*}} : tensor<32768xf32> to tensor<1x1024x32xf32>
// CHECK_SCALE-NEXT: %[[scale:.*]] = rock.transform %[[scaleRaw]] {{.*}} : tensor<1048576xf32> to tensor<1x1024x1024xf32>

// CHECK_SCALE-NEXT: rock.attention
// CHECK_SCALE-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_SCALE-NEXT: qk = elementwise otherIns(%[[scale]]
// CHECK_SCALE: softmax(qk) * %[[values]]
// CHECK_SCALE: %[[flatOutput:.*]] = rock.transform %{{.*}} {{.*}}
// CHECK_SCALE-NEXT: rock.store %[[flatOutput]] to %[[outputRaw]] by {{.*}}set
// CHECK_SCALE: return

// CHECK_SCALE-LABEL: func.func @host_naive_attention
// CHECK_SCALE: %[[qkTensor:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]
// CHECK_SCALE-DAG: %[[sqkTensor:.*]] = tosa.mul %[[qkTensor]], %[[scaleTensor:.*]], %{{.*}} : ([[squareShape]], [[squareShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[sqkTensorCast:.*]] = tosa.cast %[[sqkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[sqkTensorCast]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK_SCALE-DAG: %[[normilizedSqkTensor:.*]] = tosa.sub %[[sqkTensorCast]], %[[sqkMaxs]] : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedSqkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]
// CHECK_SCALE-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK_SCALE-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShape]], [[reducedShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} : ([[squareShape]], [[valuesShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[valuesShape]]
// CHECK_SCALE: return

// ----

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_NO_SCALE

// CHECK_NO_SCALE: module attributes {rock.arch = "[[$ARCH:.*]]"}

// CHECK_NO_SCALE-LABEL: func.func @rock_attention
// CHECK_NO_SCALE-SAME: (%[[queriesRaw:.*0]]: tensor<32768xf32>,
// CHECK_NO_SCALE-SAME: %[[keysRaw:.*1]]: tensor<32768xf32>,
// CHECK_NO_SCALE-SAME: %[[valuesRaw:.*2]]: tensor<32768xf32>,
// CHECK_NO_SCALE-SAME: %[[outputRaw:.*3]]: tensor<32768xf32>)
// CHECK_NO_SCALE-SAME: attributes {rock.arch = "[[$ARCH]]", rock.kernel}
// CHECK_NO_SCALE-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : tensor<32768xf32> to tensor<1x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[keys:.*]] = rock.transform %[[keysRaw]] {{.*}} : tensor<32768xf32> to tensor<1x32x1024xf32>
// CHECK_NO_SCALE-NEXT: %[[values:.*]] = rock.transform %[[valuesRaw]] {{.*}} : tensor<32768xf32> to tensor<1x1024x32xf32>

// CHECK_NO_SCALE-NEXT: rock.attention
// CHECK_NO_SCALE-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_NO_SCALE: softmax(qk) * %[[values]]
// CHECK_NO_SCALE: %[[flatOutput:.*]] = rock.transform %{{.*}} {{.*}}
// CHECK_NO_SCALE-NEXT: rock.store %[[flatOutput]] to %[[outputRaw]] by {{.*}}set
// CHECK_NO_SCALE: return

// CHECK_NO_SCALE-LABEL: func.func @host_naive_attention
// CHECK_NO_SCALE: %[[qkTensor:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]], %{{.*}}, %{{.*}} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]
// CHECK_NO_SCALE: %[[qkTensorCast:.*]] = tosa.cast %[[qkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[qkTensorCast]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK_NO_SCALE-DAG: %[[normilizedQkTensor:.*]] = tosa.sub %[[qkTensorCast]], %[[sqkMaxs]] : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedQkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]
// CHECK_NO_SCALE-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK_NO_SCALE-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShape]], [[reducedShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} : ([[squareShape]], [[valuesShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[valuesShape]]
// CHECK_NO_SCALE: return

// ----

// Per-tensor transpose flags. The baseline (no flag) layout is:
//   Q in [seq_q, head_qk], K in [head_qk, seq_k],
//   V in [seq_k, head_v],  O in [seq_q, head_v].
// `-transQ` / `-transV` flip the trailing two dims of the corresponding
// operand and surface a `tr` modifier on the matmul; `-transO` flips the
// output and sets the `oTransposed` attribute on `rock.attention`.

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 128 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 64 -t f16 -transQ | FileCheck %s --enable-var-scope --check-prefix=TRANS_Q
// TRANS_Q-LABEL: func.func @rock_attention
// Q (flat 4096) becomes head_qk x seq_q instead of seq_q x head_qk.
// TRANS_Q: rock.transform %{{.*}} : tensor<4096xf16> to tensor<1x32x128xf16>
// TRANS_Q: rock.transform %{{.*}} : tensor<8192xf16> to tensor<1x32x256xf16>
// TRANS_Q: rock.transform %{{.*}} : tensor<16384xf16> to tensor<1x256x64xf16>
// TRANS_Q: rock.attention
// TRANS_Q: qk = tr %{{.*}} * %{{.*}} : tensor<1x32x128xf16>, tensor<1x32x256xf16>

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 128 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 64 -t f16 -transV | FileCheck %s --enable-var-scope --check-prefix=TRANS_V
// TRANS_V-LABEL: func.func @rock_attention
// V (flat 16384) becomes head_v x seq_k instead of seq_k x head_v.
// TRANS_V: rock.transform %{{.*}} : tensor<4096xf16> to tensor<1x128x32xf16>
// TRANS_V: rock.transform %{{.*}} : tensor<8192xf16> to tensor<1x32x256xf16>
// TRANS_V: rock.transform %{{.*}} : tensor<16384xf16> to tensor<1x64x256xf16>
// TRANS_V: rock.attention
// TRANS_V: softmax(qk) * tr %{{.*}} : tensor<1x64x256xf16>

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 128 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 64 -t f16 -transO | FileCheck %s --enable-var-scope --check-prefix=TRANS_O
// TRANS_O-LABEL: func.func @rock_attention
// TRANS_O: rock.attention
// TRANS_O: oTransposed
// TRANS_O: -> tensor<1x64x128xf16>
