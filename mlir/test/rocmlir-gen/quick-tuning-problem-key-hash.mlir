// The per-problem quick-tuning maps are keyed on this hash, and this flag is
// the only place it is spelled. Changing the fields that make up the key
// silently invalidates every shipped shard, so the values below are pinned:
// if one of them changes, regenerate the maps under
// mlir/include/mlir/Dialect/Rock/Tuning/QuickTuningProblemMap.

// RUN: rocmlir-gen --arch gfx942 --operation gemm -p --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=GEMM
// GEMM: 10439942300753512616

// RUN: rocmlir-gen --arch gfx942 --operation conv -p --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=CONV_FWD
// CONV_FWD: 17009370425126842901

// Forward and backward-data are separate problems.
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_data -p --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=CONV_BWD_DATA
// CONV_BWD_DATA: 9497099527036308506

// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 1 --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=ATTN
// ATTN: 10457287879272258229

// Fusion-shaped fields are part of attention's identity.
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 1 -causal --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=ATTN_CAUSAL
// ATTN_CAUSAL: 4994105176242944686

// Transposes are part of the problem's identity.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -p -transA --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=GEMM_TRANSA
// GEMM_TRANSA: 8165922487442142896

// The architecture and the data type are carried by the lookup key that selects
// the shard, so they must not appear in the hash as well.
// RUN: rocmlir-gen --arch gfx1101 --operation gemm -p -t f16 --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=GEMM

// The conv key holds the products ci*gi and k*g, so it cannot tell a grouped
// problem from an ungrouped one with the same totals: both get the same entry.
// RUN: rocmlir-gen --arch gfx942 --operation conv -t f16 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 256 --in_h 20 --in_w 20 --out_channels 256 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 3 --padding_w 3 --groupsize 1 --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=CONV_GROUPED
// RUN: rocmlir-gen --arch gfx942 --operation conv -t f16 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 256 --in_h 20 --in_w 20 --out_channels 256 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 3 --padding_w 3 --groupsize 128 --emit-quick-tuning-problem-key-hash | FileCheck %s --check-prefix=CONV_GROUPED
// CONV_GROUPED: 18267882488113781532
