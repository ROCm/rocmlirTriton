// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -current_seq_len=33 -return_lse -split_kv 8 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv | rocmlir-opt | FileCheck %s --enable-var-scope

// Use a perf_config with mPerBlockG0 != nPerBlockG0 so the split-KV finalization
// must use the key-sequence block size rather than the query block size.
// RUN: rocmlir-gen --arch gfx942 --operation attention -t f16 -g 5 -seq_len_q 1 -seq_len_k 331 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 69 -head_dim_v 208 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=True -transO=False -causal=False -return_lse=True -split_kv=8 --perf_config=attn:v1:128,64,32,2,1,16,32,1,1,4,4 --current_seq_len=255,18,268,69,317 -pv | rocmlir-opt | FileCheck %s --check-prefix=VALID-SPLIT-KV

// CHECK: module attributes {rock.arch = "[[$ARCH:.*]]"}

// VALID-SPLIT-KV-LABEL: func.func @rock_attention_gpu
// With current_seq_len=255,18,268,69,317 and nPerBlockG0=64, the valid split
// mask is [4, 1, 5, 2, 5]. Using mPerBlockG0=128 would produce [2, 1, 3, 1, 3].
// VALID-SPLIT-KV: "tosa.const"() <{values = dense<{{\[+}}4{{\]+}}, {{\[+}}1{{\]+}}, {{\[+}}5{{\]+}}, {{\[+}}2{{\]+}}, {{\[+}}5{{\]+}}> : tensor<5x1x1x1xi32>}>

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: tensor<128xf32>,
// CHECK-SAME: %[[keysRaw:.*1]]: tensor<65536xf32>,
// CHECK-SAME: %[[valuesRaw:.*2]]: tensor<65536xf32>,
// CHECK-SAME: %[[currentSeqLenRaw:.*3]]: tensor<1xi32>,
// CHECK-SAME: %[[lseRaw:.*4]]: tensor<32xf32>,
// CHECK-SAME: %[[outputRaw:.*5]]: tensor<1024xf32>)
// CHECK-SAME: attributes {rock.arch = "[[$ARCH]]", rock.kernel}
// CHECK-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : tensor<128xf32> to tensor<4x1x32xf32>
// CHECK-NEXT: %[[keys:.*]] = rock.transform %[[keysRaw]] {{.*}} : tensor<65536xf32> to tensor<2x32x1024xf32>
// CHECK-NEXT: %[[values:.*]] = rock.transform %[[valuesRaw]] {{.*}} : tensor<65536xf32> to tensor<2x1024x32xf32>
// CHECK-NEXT: %[[currentSeqLen:.*]] = rock.transform %[[currentSeqLenRaw]] {{.*}} : tensor<1xi32> to tensor<1xi32>
// CHECK-NEXT: %[[currentSeqLenAddDim:.*]] = rock.transform %[[currentSeqLen]] {{.*}} : tensor<1xi32> to tensor<1x1xi32>
// CHECK-NEXT: %[[currentSeqLenBroadcast:.*]] = rock.transform %[[currentSeqLenAddDim]] {{.*}} : tensor<1x1xi32> to tensor<1x4xi32>
// CHECK-NEXT: %[[currentSeqLenMerge:.*]] = rock.transform %[[currentSeqLenBroadcast]] {{.*}} : tensor<1x4xi32> to tensor<4xi32>

// CHECK-NEXT: %[[output:.*]], %[[lse:.*]] = rock.attention
// CHECK-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK-NEXT: currentSeqLen = (%[[currentSeqLenMerge]] : tensor<4xi32>)
// CHECK: softmax(qk) * %[[values]]
// CHECK-NEXT: numHeadsKV = 2 : i32, numHeadsQ = 4 : i32
// CHECK-SAME: splitKV = 8 : i32
// CHECK: %[[flatOutput:.*]] = rock.transform %{{.*}} {{.*}}
// CHECK-NEXT: rock.store %[[flatOutput]] to %[[outputRaw]] by {{.*}}set
// CHECK-NEXT: %[[flatLse:.*]] = rock.transform %{{.*}} {{.*}}
// CHECK-NEXT: rock.store %[[flatLse]] to %[[lseRaw]] by {{.*}}set
// CHECK: return

// CHECK-LABEL: func.func @host_naive_attention
// CHECK: %[[keysExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 32, 1024] : tensor<2x32x1024xf32> into tensor<2x1x32x1024xf32>
// CHECK: %[[keysMul:.*]] = tosa.mul %[[keysExpanded]], %{{.*}}, %{{.*}} : (tensor<2x1x32x1024xf32>, tensor<2x2x32x1024xf32>, tensor<1xi8>) -> tensor<2x2x32x1024xf32>
// CHECK: %[[keysTensor:.*]] = tensor.collapse_shape %[[keysMul]] {{.*}} : tensor<2x2x32x1024xf32> into tensor<4x32x1024xf32>
// CHECK: %[[valuesExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 1024, 32] : tensor<2x1024x32xf32> into tensor<2x1x1024x32xf32>
// CHECK: %[[valuesMul:.*]] = tosa.mul %[[valuesExpanded]], %{{.*}}, %{{.*}} : (tensor<2x1x1024x32xf32>, tensor<2x2x1024x32xf32>, tensor<1xi8>) -> tensor<2x2x1024x32xf32>
// CHECK: %[[valuesTensor:.*]] = tensor.collapse_shape %[[valuesMul]] {{.*}} : tensor<2x2x1024x32xf32> into [[valuesShape:tensor<.*>]]
// CHECK: %[[qkTensorOrig:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]

// CHECK: %[[currSeqLenTensorDumbReshaped:.*]] = tosa.reshape %[[currSeqLenTensor:.*]], %{{.*}} : (tensor<1xi32>, !tosa.shape<1>) -> tensor<1xi32>
// CHECK: %[[currSeqLenTensorReshaped:.*]] = tosa.reshape %[[currSeqLenTensorDumbReshaped]], %{{.*}} : (tensor<1xi32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
// CHECK: %[[qkTensorCasted:.*]] = tosa.cast %[[qkTensorOrig]] : (tensor<4x1x1024xf32>) -> tensor<4x1x1024xf32>
// CHECK: %[[qkTensorReshaped:.*]] = tosa.reshape %[[qkTensorCasted]], %{{.*}} : (tensor<4x1x1024xf32>, !tosa.shape<4>) -> tensor<1x4x1x1024xf32>
// CHECK: %[[range:.*]] = "tosa.const"() <{values = {{.*}} : tensor<1024xi32>}> : () -> tensor<1024xi32>
// CHECK: %[[rangeReshaped:.*]] = tosa.reshape %[[range]], %{{.*}} : (tensor<1024xi32>, !tosa.shape<4>) -> tensor<1x1x1x1024xi32>
// CHECK: %[[one:.*]] = "tosa.const"() <{values = dense<1> : tensor<1x4x1x1024xi32>}> : () -> tensor<1x4x1x1024xi32>
// CHECK: %[[rangeBroadcast:.*]] = tosa.mul %[[rangeReshaped]], %[[one]], %{{.*}} : (tensor<1x1x1x1024xi32>, tensor<1x4x1x1024xi32>, tensor<1xi8>) -> tensor<1x4x1x1024xi32>
// CHECK: %[[one2:.*]] = "tosa.const"() <{values = dense<1> : tensor<1x4x1x1024xi32>}> : () -> tensor<1x4x1x1024xi32>
// CHECK: %[[currSeqLenTensorBroadcast:.*]] = tosa.mul %[[currSeqLenTensorReshaped]], %[[one2]], %{{.*}} : (tensor<1x1x1x1xi32>, tensor<1x4x1x1024xi32>, tensor<1xi8>) -> tensor<1x4x1x1024xi32>
// CHECK: %[[mask:.*]] = tosa.greater %[[rangeBroadcast]], %[[currSeqLenTensorBroadcast]] : (tensor<1x4x1x1024xi32>, tensor<1x4x1x1024xi32>) -> tensor<1x4x1x1024xi1>
// CHECK: %[[negInf:.*]] = "tosa.const"() <{values = dense<0xFF800000> : tensor<1x4x1x1024xf32>}> : () -> tensor<1x4x1x1024xf32>
// CHECK: %[[qkTensorBeforeReshape:.*]] = tosa.select %[[mask]], %[[negInf]], %[[qkTensorReshaped]] : (tensor<1x4x1x1024xi1>, tensor<1x4x1x1024xf32>, tensor<1x4x1x1024xf32>) -> tensor<1x4x1x1024xf32>
// CHECK: %[[qkTensor:.*]] = tosa.reshape %[[qkTensorBeforeReshape]], %{{.*}} : (tensor<1x4x1x1024xf32>, !tosa.shape<3>) -> tensor<4x1x1024xf32>

// CHECK-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[qkTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK-DAG: %[[normilizedQkTensor:.*]] = tosa.sub %[[qkTensor]], %{{.*}} : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedQkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]

// CHECK-DAG: %[[expsSumsTensorCasted:.*]] = tosa.cast %[[expsSumsTensor]] : (tensor<4x1x1xf32>) -> tensor<4x1x1xf32>
// CHECK-DAG: %[[sqkMaxsCasted:.*]] = tosa.cast %[[sqkMaxs]] : (tensor<4x1x1xf32>) -> tensor<4x1x1xf32>
// CHECK-DAG: %[[logL:.*]] = tosa.log %[[expsSumsTensorCasted]] : (tensor<4x1x1xf32>) -> tensor<4x1x1xf32>
// CHECK-DAG: %[[resultLse:.*]] = tosa.add %[[logL]], %[[sqkMaxsCasted]] : (tensor<4x1x1xf32>, tensor<4x1x1xf32>) -> tensor<4x1x1xf32>

// CHECK-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %{{.*}} : ([[reducedShape]]) -> [[reducedShape]]
// CHECK-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShape]], [[reducedShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK-DAG: %[[softmaxTensorCasted:.*]] = tosa.cast %[[softmaxTensor]] : (tensor<4x1x1024xf32>) -> tensor<4x1x1024xf32>
// CHECK-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCasted]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : (tensor<4x1x1024xf32>, tensor<4x1024x32xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x1x32xf32>
// CHECK: %[[expandNumHeads:.*]] = tensor.expand_shape %[[resultTensor]] {{.*}} : tensor<4x1x32xf32> into tensor<4x1x1x32xf32>
// CHECK: %[[onePad:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<4x8x1x32xf32>}> : () -> tensor<4x8x1x32xf32>
// CHECK: %[[broadcastSplitKV:.*]] = tosa.mul %[[expandNumHeads]], %[[onePad]], %{{.*}} : (tensor<4x1x1x32xf32>, tensor<4x8x1x32xf32>, tensor<1xi8>) -> tensor<4x8x1x32xf32>
// CHECK: %[[collapseToBatch:.*]] = tensor.collapse_shape %[[broadcastSplitKV]] {{.*}} : tensor<4x8x1x32xf32> into tensor<32x1x32xf32>
// CHECK: tosa.reshape %[[collapseToBatch]], %{{.*}} : (tensor<32x1x32xf32>, !tosa.shape<1>) -> tensor<1024xf32>
// CHECK: return
