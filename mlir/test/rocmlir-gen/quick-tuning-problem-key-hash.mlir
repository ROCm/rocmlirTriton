// Each pair below pins a problem hash followed by the hash of the ordered field
// names used to build it. The latter is the generated map's automatic version:
// adding, removing, renaming, or reordering a lookup-key field changes it
// without a manual version bump. Regenerate the maps under
// mlir/include/mlir/Dialect/Rock/Tuning/QuickTuningProblemMap when it changes.

// RUN: rocmlir-gen --arch gfx942 --operation gemm -p --emit-quick-tuning-problem-key-hash --emit-quick-tuning-table-lookup-key-version-hash | FileCheck %s --check-prefix=GEMM_VERSION
// GEMM_VERSION: 10439942300753512616
// GEMM_VERSION-NEXT: 6791176183107838810

// RUN: rocmlir-gen --arch gfx942 --operation conv -p --emit-quick-tuning-problem-key-hash --emit-quick-tuning-table-lookup-key-version-hash | FileCheck %s --check-prefix=CONV_FWD
// CONV_FWD: 17009370425126842901
// CONV_FWD-NEXT: 12815695606661592525

// Forward and backward-data are separate problems under the same schema.
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_data -p --emit-quick-tuning-problem-key-hash --emit-quick-tuning-table-lookup-key-version-hash | FileCheck %s --check-prefix=CONV_BWD_DATA
// CONV_BWD_DATA: 9497099527036308506
// CONV_BWD_DATA-NEXT: 12815695606661592525

// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 1 --emit-quick-tuning-problem-key-hash --emit-quick-tuning-table-lookup-key-version-hash | FileCheck %s --check-prefix=ATTN_VERSION
// ATTN_VERSION: 10457287879272258229
// ATTN_VERSION-NEXT: 2158629286767709275

// RUN: rocmlir-gen --arch gfx942 --operation gemm_gemm -t f32 -p --emit-quick-tuning-problem-key-hash --emit-quick-tuning-table-lookup-key-version-hash | FileCheck %s --check-prefix=GEMM_GEMM_VERSION
// GEMM_GEMM_VERSION: 6461224018699686721
// GEMM_GEMM_VERSION-NEXT: 12034527386330093365

// RUN: rocmlir-gen --arch gfx942 --operation conv_gemm -t f32 -p --emit-quick-tuning-problem-key-hash --emit-quick-tuning-table-lookup-key-version-hash | FileCheck %s --check-prefix=CONV_GEMM_VERSION
// CONV_GEMM_VERSION: 916381620957689987
// CONV_GEMM_VERSION-NEXT: 10231796635899358662

// Fusion-shaped fields are part of attention's identity.
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 1 -causal --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=ATTN_CAUSAL
// ATTN_CAUSAL: 4994105176242944686

// Transposes are part of the problem's identity.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -p -transA --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=GEMM_TRANSA
// GEMM_TRANSA: 8165922487442142896

// The architecture and the data type are carried by the lookup key that selects
// the shard, so they must not appear in the hash as well.
// RUN: rocmlir-gen --arch gfx1101 --operation gemm -p -t f16 --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=GEMM_ARCH
// GEMM_ARCH: 10439942300753512616

//===----------------------------------------------------------------------===//
// Field significance
//===----------------------------------------------------------------------===//
// The problem and version hashes pinned above catch changes to key values and
// field names. These checks document field significance too: dropping a field
// the key needs makes two problems share a row, so one is served a ranking
// measured on the other, while keeping a field it does not need splits one
// problem into several that the shards have no entry for.
//
// The reference is getQuickTuningProblemKey. It is separate from the tuning DB
// key MIGraphX reads through mlirRockTuningGetKey, and omits what the
// <arch, kernel, data type> map key already carries and the machine the
// measurement ran on.

// RUN: rocmlir-gen --arch gfx942 --operation gemm -p --emit-quick-tuning-problem-key-hash > %t.gemm

// Dropped: compute-unit and chiplet counts name the machine, not the problem,
// and the lookup key is keyed on the architecture alone -- as the set cover
// always has been -- so a second SKU of one chip must not split the row.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -p --num_cu 64 --emit-quick-tuning-problem-key-hash | diff - %t.gemm
// RUN: rocmlir-gen --arch gfx942 --operation gemm -p --num_chiplets 2 --emit-quick-tuning-problem-key-hash | diff - %t.gemm

// The input element type lives in the outer table key. A different output type
// was not represented in the exhaustive results used by the shipped maps, so
// it must not alias the base problem.
// RUN: not rocmlir-gen --arch gfx942 --operation gemm -p -t f16 -out_datatype f32 --emit-quick-tuning-problem-key-hash 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-OUTPUT-TYPE
// UNSUPPORTED-OUTPUT-TYPE: fields not represented by the shipped maps: output_data_type

// Kept: every transpose. -transA is pinned above; these are its siblings,
// which a truncated field list would silently collapse onto the base problem.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -p -transB --emit-quick-tuning-problem-key-hash | not diff - %t.gemm
// RUN: rocmlir-gen --arch gfx942 --operation gemm -p -transO --emit-quick-tuning-problem-key-hash | not diff - %t.gemm

// Kept: a convolution's stride, dilation and padding. These do not change the
// tensor shapes, so a key built from shapes alone would collapse all four
// convolutions below onto one row, but they do change the gemm the problem
// lowers to and therefore what tunes well for it.
// DEFINE: %{conv} = rocmlir-gen --arch gfx942 --operation conv -t f32 --batchsize=1 --in_channels=8 --in_h=8 --in_w=8 --out_channels=8 --fil_h=3 --fil_w=3 --padding_h=1
// RUN: %{conv} --padding_w=1 --emit-quick-tuning-problem-key-hash > %t.conv
// RUN: %{conv} --padding_w=1 --num_cu=64 --emit-quick-tuning-problem-key-hash | diff - %t.conv
// RUN: %{conv} --padding_w=1 --conv_stride_h=2 --emit-quick-tuning-problem-key-hash | not diff - %t.conv
// RUN: %{conv} --padding_w=1 --dilation_w=2 --emit-quick-tuning-problem-key-hash | not diff - %t.conv
// RUN: %{conv} --padding_w=2 --emit-quick-tuning-problem-key-hash | not diff - %t.conv

// Kept: attention's head counts, and the pre-softmax scale and bias. The last
// two are not operands but ops fused into the attention body, which
// getAttentionScaleBias recovers by walking it -- so unlike conv and gemm, an
// attention problem's identity does depend on what was fused into it.
// DEFINE: %{attn} = rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 1
// RUN: %{attn} --num_cu=64 --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=ATTN
// ATTN: 10457287879272258229
// RUN: %{attn} --emit-quick-tuning-problem-key-hash > %t.attn
// RUN: %{attn} -num_heads_q 2 -num_heads_kv 1 --emit-quick-tuning-problem-key-hash | not diff - %t.attn
// RUN: %{attn} -with-attn-scale --emit-quick-tuning-problem-key-hash | not diff - %t.attn
// RUN: %{attn} -with-attn-bias --emit-quick-tuning-problem-key-hash | not diff - %t.attn
// RUN: not %{attn} -softmax_dtype f16 --emit-quick-tuning-problem-key-hash 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-SOFTMAX-TYPE
// UNSUPPORTED-SOFTMAX-TYPE: fields not represented by the shipped maps: softmax_data_type
// RUN: not %{attn} -last_valid_kv_index 511 --emit-quick-tuning-problem-key-hash 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-KV-CACHE
// UNSUPPORTED-KV-CACHE: fields not represented by the shipped maps: last_valid_kv_index

// Existing maps were measured as ungrouped problems. The key still holds the
// total channel products, but a grouped runtime problem must fall back instead
// of aliasing an ungrouped ranking.
// RUN: rocmlir-gen --arch gfx942 --operation conv -t f16 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 256 --in_h 20 --in_w 20 --out_channels 256 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 3 --padding_w 3 --groupsize 1 --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=CONV_GROUPED
// RUN: not rocmlir-gen --arch gfx942 --operation conv -t f16 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 256 --in_h 20 --in_w 20 --out_channels 256 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 3 --padding_w 3 --groupsize 128 --emit-quick-tuning-problem-key-hash 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-CONV-GROUPS
// CONV_GROUPED: 18267882488113781532
// UNSUPPORTED-CONV-GROUPS: fields not represented by the shipped maps: convolution_groups

// A newly attached compiler field is fail-closed until it is deliberately
// added to the problem key or classified as irrelevant.
// RUN: not rocmlir-gen --emit-quick-tuning-problem-key-hash %s 2>&1 | FileCheck %s --check-prefix=UNCLASSIFIED-FIELD
// UNCLASSIFIED-FIELD: fields not represented by the shipped maps: attribute:futureTuningFlag

module {
  func.func @unclassified_field(%a: tensor<1x16x16xf32>,
                                %b: tensor<1x16x16xf32>)
      -> tensor<1x16x16xf32>
      attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel} {
    %result = rock.gemm %a * %b {futureTuningFlag = true}
      : tensor<1x16x16xf32> * tensor<1x16x16xf32>
        -> tensor<1x16x16xf32>
    return %result : tensor<1x16x16xf32>
  }
}
