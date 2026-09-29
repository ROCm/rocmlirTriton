// Asymmetric padding makes the input tensor's `Pad` transform produce lower
// coordinates on both sides of the valid range: `ho * stride + y * dilation -
// leftPad` is negative for taps that fall into the left padding, and `>= inH`
// for taps in the right padding. `updateValidityAfter` masks both with a
// single `arith.cmpi ult` against the dimension bound, which rejects negatives
// because they wrap to large unsigned values. Since the offset of a masked
// element is still computed from the negative coordinate (and fed to
// `arith.divui` / `arith.remui` further down the chain), a regression in that
// masking shows up as garbage accumulated from the padding region rather than
// as a fault.
//
// The first config below is the deterministic gfx1100 correctness failure from
// AIROCMLIR-708 / https://github.com/ROCm/rocMLIR/pull/2353, which needs the
// negative (left-padding) half of the check. The second mirrors the padding
// onto the right so the same shape also covers the `>= bound` half.

// RUN: rocmlir-gen -pv --operation conv -t f16 --arch %arch --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 64 --in_channels 64 --in_h 4 --in_w 4 --out_channels 64 --fil_h 2 --fil_w 2 --dilation_h 1 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 1 --padding_h_l 3 --padding_h_r 0 --padding_w_l 2 --padding_w_r 0 \
// RUN:   | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]

// RUN: rocmlir-gen -pv --operation conv -t f16 --arch %arch --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 64 --in_channels 64 --in_h 4 --in_w 4 --out_channels 64 --fil_h 2 --fil_w 2 --dilation_h 1 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 1 --padding_h_l 0 --padding_h_r 3 --padding_w_l 0 --padding_w_r 2 \
// RUN:   | rocmlir-driver -c | rocm-run | FileCheck %s --check-prefix=RIGHT

// RIGHT: [1 1 1]
