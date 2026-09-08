// On gfx11 before gfx1150 (FeatureDPPSrc1SGPR) a DPP src1 must be a VGPR --
// neither an SGPR nor an immediate. LLVM used to enforce only the register half
// of that rule, so GCNDPPCombine would fold the softmax sum's v_mov_b32_dpp
// lane shuffle into a v_add_f16 whose other operand is the inline constant 0
// (the identity of the sum), emitting an unencodable
//
//     v_add_f16_e64_dpp v40, v1, 0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
//
// that the assembler rejects with "src1 immediate operand invalid for
// instruction", which then cascades into an LLD failure and "Lowering failed".
// Fixed by llvm-patches/patch201494.patch (upstream llvm/llvm-project#201494).
//
// The fold only happens when the reduction identity survives as an inline
// constant, which makes the failure tile- and shape-dependent: it first showed
// up as mixr-attention-lse-broadcasts.mlir breaking when the gfx1100 f16
// attention quick-tuning list started picking mPerBlockG0=64/kPerBlock=64 over
// the old conservative 32,32,32 default. The shape below is that test's, and
// the tile is pinned so this reproducer stays independent of the tuning list.
// An f16 softmax (v_add_f16) and the LSE output are both required to trigger it.

// RUN: rocmlir-gen --arch %arch --operation attention -t f16 --softmax_dtype f16 -g 2 -seq_len_q 5 -seq_len_k 5 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 64 -head_dim_v 64 -with-attn-scale=False -with-attn-bias=True -transQ=False -transK=True -transV=False -transO=False -causal=False -return_lse=True -split_kv=1 --perf_config=attn:v1:64,32,64,1,1,4,0,1,2,0,0 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
// CHECK: [1 1 1]
