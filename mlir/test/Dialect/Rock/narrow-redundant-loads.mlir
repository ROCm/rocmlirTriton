// Unit tests for the rocmlirTriton rock-narrow-redundant-loads pass.
//
// The pass runs on TTIR, inside makeTTIR right after triton-combine-ops. It
// looks for a tt.load whose result Triton's alignment analysis proves constant
// across a whole tensor dimension, and rewrites it into a load of the index-0
// slice along that dimension plus a tt.broadcast back to the original shape.
// A load it cannot prove redundant, or whose address it cannot re-materialize
// at the narrower shape, is left exactly as it was.

// RUN: rocmlir-opt -rock-narrow-redundant-loads --split-input-file %s | FileCheck %s

// The group-quantized GEMM scale: the scale index divides the K coordinate by
// the group size, so a K tile that lands inside one group reads the same scale
// for all of its K positions. Only dim 0 is redundant; the N coordinate still
// picks a different scale per lane.

// CHECK-LABEL: @group_quant_scale
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
// CHECK-NEXT:   tt.broadcast %[[VAL]] : tensor<1x128xf16> -> tensor<32x128xf16>
func.func @group_quant_scale(%base: !tt.ptr<f16>, %kBlock: i32) -> tensor<32x128xf16> {
  %kPerBlock = arith.constant 32 : i32
  %groupSize = arith.constant dense<128> : tensor<32x1xi32>
  %kRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %kBase = arith.muli %kBlock, %kPerBlock : i32
  %kBaseT = tt.splat %kBase : i32 -> tensor<32x1xi32>
  %k = arith.addi %kBaseT, %kCol : tensor<32x1xi32>
  %group = arith.divui %k, %groupSize : tensor<32x1xi32>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %groupB = tt.broadcast %group : tensor<32x1xi32> -> tensor<32x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %off = arith.addi %nB, %groupB : tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// The same kernel tuned with a K tile four times the group size. The tile now
// spans four groups, so the scale is only constant across 128 of the tile's 512
// K positions and the load has to stay as it is. The rewrite is therefore
// self-limiting under tuning rather than needing a guard against large tiles.

// CHECK-LABEL: @k_tile_straddles_groups
//  CHECK-NOT:   tt.broadcast {{.*}} -> tensor<512x128xf16>
//      CHECK:   tt.load %{{.*}} : tensor<512x128x!tt.ptr<f16>>
func.func @k_tile_straddles_groups(%base: !tt.ptr<f16>, %kBlock: i32) -> tensor<512x128xf16> {
  %kPerBlock = arith.constant 512 : i32
  %groupSize = arith.constant dense<128> : tensor<512x1xi32>
  %kRange = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
  %kBase = arith.muli %kBlock, %kPerBlock : i32
  %kBaseT = tt.splat %kBase : i32 -> tensor<512x1xi32>
  %k = arith.addi %kBaseT, %kCol : tensor<512x1xi32>
  %group = arith.divui %k, %groupSize : tensor<512x1xi32>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %groupB = tt.broadcast %group : tensor<512x1xi32> -> tensor<512x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<512x128xi32>
  %off = arith.addi %nB, %groupB : tensor<512x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<512x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<512x128x!tt.ptr<f16>>, tensor<512x128xi32>
  %val = tt.load %addr : tensor<512x128x!tt.ptr<f16>>
  return %val : tensor<512x128xf16>
}

// -----

// The same scale read with the group index on the other axis. Which axis is
// redundant follows the address, not the position in the tile.

// CHECK-LABEL: @uniform_along_dim_one
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}} : tensor<32x1x!tt.ptr<f16>>
// CHECK-NEXT:   tt.broadcast %[[VAL]] : tensor<32x1xf16> -> tensor<32x128xf16>
func.func @uniform_along_dim_one(%base: !tt.ptr<f16>) -> tensor<32x128xf16> {
  %mRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %mCol = tt.expand_dims %mRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %off = tt.broadcast %mCol : tensor<32x1xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// A dequantizing GEMM reads both a scale and a zero point per group. Each
// redundant load is narrowed on its own.

// CHECK-LABEL: @two_redundant_loads
//      CHECK:   %[[SCALE:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
// CHECK-NEXT:   %[[SCALE_FULL:.*]] = tt.broadcast %[[SCALE]] : tensor<1x128xf16> -> tensor<32x128xf16>
//      CHECK:   %[[ZERO:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
// CHECK-NEXT:   %[[ZERO_FULL:.*]] = tt.broadcast %[[ZERO]] : tensor<1x128xf16> -> tensor<32x128xf16>
//      CHECK:   arith.subf %[[SCALE_FULL]], %[[ZERO_FULL]]
func.func @two_redundant_loads(%scaleBase: !tt.ptr<f16>, %zeroBase: !tt.ptr<f16>) -> tensor<32x128xf16> {
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %off = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %scalePtrs = tt.splat %scaleBase : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %scaleAddr = tt.addptr %scalePtrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %scale = tt.load %scaleAddr : tensor<32x128x!tt.ptr<f16>>
  %zeroPtrs = tt.splat %zeroBase : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %zeroAddr = tt.addptr %zeroPtrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %zero = tt.load %zeroAddr : tensor<32x128x!tt.ptr<f16>>
  %res = arith.subf %scale, %zero : tensor<32x128xf16>
  return %res : tensor<32x128xf16>
}

// -----

// The element type plays no part in the decision: what is narrowed is the
// address, and the fill and any conversion of the loaded values stay on the
// full tile.

// CHECK-LABEL: @narrow_integer_load
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<i8>>
// CHECK-NEXT:   %[[FULL:.*]] = tt.broadcast %[[VAL]] : tensor<1x128xi8> -> tensor<32x128xi8>
//      CHECK:   arith.sitofp %[[FULL]]
func.func @narrow_integer_load(%base: !tt.ptr<i8>) -> tensor<32x128xf16> {
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %off = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<32x128x!tt.ptr<i8>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<i8>>, tensor<32x128xi32>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<i8>>
  %res = arith.sitofp %val : tensor<32x128xi8> to tensor<32x128xf16>
  return %res : tensor<32x128xf16>
}

// -----

// Packed group indices arrive through bit arithmetic rather than a division.
// Elementwise arithmetic is re-materialized at the narrow shape by cloning it
// over narrowed operands.

// CHECK-LABEL: @address_through_arithmetic
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
// CHECK-NEXT:   tt.broadcast %[[VAL]] : tensor<1x128xf16> -> tensor<32x128xf16>
func.func @address_through_arithmetic(%base: !tt.ptr<f16>, %packed: i32) -> tensor<32x128xf16> {
  %maskBits = arith.constant dense<255> : tensor<1x128xi32>
  %shift = arith.constant dense<8> : tensor<1x128xi32>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %packedT = tt.splat %packed : i32 -> tensor<1x128xi32>
  %biased = arith.addi %packedT, %nRow : tensor<1x128xi32>
  %shifted = arith.shrui %biased, %shift : tensor<1x128xi32>
  %group = arith.andi %shifted, %maskBits : tensor<1x128xi32>
  %off = tt.broadcast %group : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// A load off a bare splat reads one address for the whole tile, so every
// dimension is redundant and the tile collapses to a single element.

// CHECK-LABEL: @fully_uniform
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x1x!tt.ptr<f16>>
// CHECK-NEXT:   tt.broadcast %[[VAL]] : tensor<1x1xf16> -> tensor<16x64xf16>
func.func @fully_uniform(%base: !tt.ptr<f16>) -> tensor<16x64xf16> {
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
  %val = tt.load %ptrs : tensor<16x64x!tt.ptr<f16>>
  return %val : tensor<16x64xf16>
}

// -----

// The same collapse to a single element, but reached through the address
// computation rather than off a bare splat: a group size as large as the whole K
// tile makes dim 0 redundant, and the broadcast over N makes dim 1 redundant.
// Narrowing both at once is what makes tt.expand_dims drop its axis while the
// dimension it expands is itself collapsing.

// CHECK-LABEL: @uniform_along_both_dims
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x1x!tt.ptr<f16>>
// CHECK-NEXT:   tt.broadcast %[[VAL]] : tensor<1x1xf16> -> tensor<32x128xf16>
func.func @uniform_along_both_dims(%base: !tt.ptr<f16>) -> tensor<32x128xf16> {
  %groupSize = arith.constant dense<32> : tensor<32x1xi32>
  %kRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %group = arith.divui %kCol, %groupSize : tensor<32x1xi32>
  %off = tt.broadcast %group : tensor<32x1xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// An ordinary tiled load reads a distinct element per lane along both dims.

// CHECK-LABEL: @contiguous_tile
//  CHECK-NOT:   tt.broadcast {{.*}} -> tensor<32x128xf16>
//      CHECK:   tt.load %{{.*}} : tensor<32x128x!tt.ptr<f16>>
func.func @contiguous_tile(%base: !tt.ptr<f16>) -> tensor<32x128xf16> {
  %stride = arith.constant dense<128> : tensor<32x1xi32>
  %mRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %mCol = tt.expand_dims %mRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row = arith.muli %mCol, %stride : tensor<32x1xi32>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %rowB = tt.broadcast %row : tensor<32x1xi32> -> tensor<32x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %off = arith.addi %rowB, %nB : tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// A mask that is itself uniform along the redundant dim narrows with the
// address, and its zero fill narrows as a splat.

// CHECK-LABEL: @masked_uniform_along_dim
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x128x!tt.ptr<f16>>
// CHECK-NEXT:   tt.broadcast %[[VAL]] : tensor<1x128xf16> -> tensor<32x128xf16>
func.func @masked_uniform_along_dim(%base: !tt.ptr<f16>, %n: i32) -> tensor<32x128xf16> {
  %zero = arith.constant dense<0.0> : tensor<32x128xf16>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %limit = tt.splat %n : i32 -> tensor<1x128xi32>
  %nMask = arith.cmpi slt, %nRow, %limit : tensor<1x128xi32>
  %mask = tt.broadcast %nMask : tensor<1x128xi1> -> tensor<32x128xi1>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %nB : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr, %mask, %zero : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// Address uniform on dim 0, mask not. Load unmasked, then select the fill back.

// CHECK-LABEL: @mask_varies_along_dim
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
// CHECK-NEXT:   %[[FULL:.*]] = tt.broadcast %[[VAL]] : tensor<1x128xf16> -> tensor<32x128xf16>
// CHECK-NEXT:   arith.select %{{.*}}, %[[FULL]], %{{.*}} : tensor<32x128xi1>, tensor<32x128xf16>
func.func @mask_varies_along_dim(%base: !tt.ptr<f16>, %k: i32) -> tensor<32x128xf16> {
  %zero = arith.constant dense<0.0> : tensor<32x128xf16>
  %kRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %limit = tt.splat %k : i32 -> tensor<32x1xi32>
  %kMask = arith.cmpi slt, %kCol, %limit : tensor<32x1xi32>
  %mask = tt.broadcast %kMask : tensor<32x1xi1> -> tensor<32x128xi1>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %nB : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr, %mask, %zero : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// The same rewrite on a load with no fill. Triton leaves the masked-out lanes
// undefined there, but the backends give them zero, so the select goes in
// against a zero constant rather than letting the loaded value reach them.

// CHECK-LABEL: @mask_varies_along_dim_no_other
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
// CHECK-NEXT:   %[[FULL:.*]] = tt.broadcast %[[VAL]] : tensor<1x128xf16> -> tensor<32x128xf16>
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<32x128xf16>
// CHECK-NEXT:   arith.select %{{.*}}, %[[FULL]], %[[ZERO]] : tensor<32x128xi1>, tensor<32x128xf16>
func.func @mask_varies_along_dim_no_other(%base: !tt.ptr<f16>, %k: i32) -> tensor<32x128xf16> {
  %kRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %limit = tt.splat %k : i32 -> tensor<32x1xi32>
  %kMask = arith.cmpi slt, %kCol, %limit : tensor<32x1xi32>
  %mask = tt.broadcast %kMask : tensor<32x1xi1> -> tensor<32x128xi1>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %nB : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr, %mask : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// The alignment analysis sees through a transpose, but the rewrite has no rule
// for one, so it declines rather than guessing. Nothing of the half-built
// narrow address is left behind.

// CHECK-LABEL: @address_not_rematerializable
//  CHECK-NOT:   tensor<1x128x!tt.ptr<f16>>
//      CHECK:   tt.load %{{.*}} : tensor<32x128x!tt.ptr<f16>>
func.func @address_not_rematerializable(%base: !tt.ptr<f16>) -> tensor<32x128xf16> {
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nCol = tt.expand_dims %nRange {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %wide = tt.broadcast %nCol : tensor<128x1xi32> -> tensor<128x32xi32>
  %off = tt.trans %wide {order = array<i32: 1, 0>} : tensor<128x32xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// Address and mask are both uniform along dim 0, but the fill is not, and the
// alignment analysis does not account for it. Broadcasting the narrow result
// would replicate the fill of the first lane, so the load stays put.

// CHECK-LABEL: @fill_varies_along_dim
//  CHECK-NOT:   tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
//      CHECK:   tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x128x!tt.ptr<f16>>
func.func @fill_varies_along_dim(%base: !tt.ptr<f16>, %n: i32) -> tensor<32x128xf16> {
  %kRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %kB = tt.broadcast %kCol : tensor<32x1xi32> -> tensor<32x128xi32>
  %fill = arith.sitofp %kB : tensor<32x128xi32> to tensor<32x128xf16>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %limit = tt.splat %n : i32 -> tensor<1x128xi32>
  %nMask = arith.cmpi slt, %nRow, %limit : tensor<1x128xi32>
  %mask = tt.broadcast %nMask : tensor<1x128xi1> -> tensor<32x128xi1>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %nB : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr, %mask, %fill : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// Per-channel bias: address repeats along N, mask is the N bounds check.

// CHECK-LABEL: @per_channel_bias
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}} : tensor<128x1x!tt.ptr<f32>>
// CHECK-NEXT:   %[[FULL:.*]] = tt.broadcast %[[VAL]] : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:   arith.select %{{.*}}, %[[FULL]], %{{.*}} : tensor<128x128xi1>, tensor<128x128xf32>
func.func @per_channel_bias(%base: !tt.ptr<f32>, %n: i32) -> tensor<128x128xf32> {
  %zero = arith.constant dense<0.0> : tensor<128x128xf32>
  %mRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %mCol = tt.expand_dims %mRange {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %chan = tt.broadcast %mCol : tensor<128x1xi32> -> tensor<128x128xi32>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %limit = tt.splat %n : i32 -> tensor<1x128xi32>
  %nMask = arith.cmpi slt, %nRow, %limit : tensor<1x128xi32>
  %mask = tt.broadcast %nMask : tensor<1x128xi1> -> tensor<128x128xi1>
  %ptrs = tt.splat %base : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  %addr = tt.addptr %ptrs, %chan : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
  %val = tt.load %addr, %mask, %zero : tensor<128x128x!tt.ptr<f32>>
  return %val : tensor<128x128xf32>
}

// -----

// The same bias load bounded in both directions, so the mask is the conjunction
// of an M check and an N check. The M check is invariant along the collapsed
// dim, so it stays on the narrowed load and keeps it from reading channels the
// original skipped; only the N check has to wait for the select.

// CHECK-LABEL: @per_channel_bias_2d_bounds_check
//      CHECK:   %[[MMASK:.*]] = arith.cmpi slt, %{{.*}} : tensor<128x1xi32>
//      CHECK:   %[[FULL:.*]] = arith.andi
//      CHECK:   %[[VAL:.*]] = tt.load %{{.*}}, %[[MMASK]] : tensor<128x1x!tt.ptr<f32>>
// CHECK-NEXT:   %[[WIDE:.*]] = tt.broadcast %[[VAL]] : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:   arith.select %[[FULL]], %[[WIDE]], %{{.*}} : tensor<128x128xi1>, tensor<128x128xf32>
func.func @per_channel_bias_2d_bounds_check(%base: !tt.ptr<f32>, %m: i32, %n: i32) -> tensor<128x128xf32> {
  %zero = arith.constant dense<0.0> : tensor<128x128xf32>
  %mRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %mCol = tt.expand_dims %mRange {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %mLimit = tt.splat %m : i32 -> tensor<128x1xi32>
  %mMaskCol = arith.cmpi slt, %mCol, %mLimit : tensor<128x1xi32>
  %mMask = tt.broadcast %mMaskCol : tensor<128x1xi1> -> tensor<128x128xi1>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %nLimit = tt.splat %n : i32 -> tensor<1x128xi32>
  %nMaskRow = arith.cmpi slt, %nRow, %nLimit : tensor<1x128xi32>
  %nMask = tt.broadcast %nMaskRow : tensor<1x128xi1> -> tensor<128x128xi1>
  %mask = arith.andi %mMask, %nMask : tensor<128x128xi1>
  %chan = tt.broadcast %mCol : tensor<128x1xi32> -> tensor<128x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  %addr = tt.addptr %ptrs, %chan : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
  %val = tt.load %addr, %mask, %zero : tensor<128x128x!tt.ptr<f32>>
  return %val : tensor<128x128xf32>
}

// -----

// A conjunct that varies along the collapsed dim *and* the surviving one is
// neither sliceable nor safe to drop, so the load stays as it is even though
// the other conjunct would have been fine on its own.

// CHECK-LABEL: @mask_conjunct_varies_along_both_dims
//  CHECK-NOT:   tt.load %{{.*}} : tensor<32x1x!tt.ptr<f16>>
//      CHECK:   tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x128x!tt.ptr<f16>>
func.func @mask_conjunct_varies_along_both_dims(%base: !tt.ptr<f16>, %m: i32, %lim: i32) -> tensor<32x128xf16> {
  %zero = arith.constant dense<0.0> : tensor<32x128xf16>
  %mRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %mCol = tt.expand_dims %mRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %mLimit = tt.splat %m : i32 -> tensor<32x1xi32>
  %mMaskCol = arith.cmpi slt, %mCol, %mLimit : tensor<32x1xi32>
  %mMask = tt.broadcast %mMaskCol : tensor<32x1xi1> -> tensor<32x128xi1>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %mB = tt.broadcast %mCol : tensor<32x1xi32> -> tensor<32x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %diag = arith.addi %mB, %nB : tensor<32x128xi32>
  %diagLimit = tt.splat %lim : i32 -> tensor<32x128xi32>
  %diagMask = arith.cmpi slt, %diag, %diagLimit : tensor<32x128xi32>
  %mask = arith.andi %mMask, %diagMask : tensor<32x128xi1>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %mB : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr, %mask, %zero : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}

// -----

// Address repeats on dim 1, but mask varies on surviving dim 0: leave alone.

// CHECK-LABEL: @mask_varies_along_surviving_dim
//  CHECK-NOT:   tt.load %{{.*}} : tensor<32x1x!tt.ptr<f16>>
//      CHECK:   tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x128x!tt.ptr<f16>>
func.func @mask_varies_along_surviving_dim(%base: !tt.ptr<f16>, %lim: i32) -> tensor<32x128xf16> {
  %zero = arith.constant dense<0.0> : tensor<32x128xf16>
  %mRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %mCol = tt.expand_dims %mRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %mB = tt.broadcast %mCol : tensor<32x1xi32> -> tensor<32x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %diag = arith.addi %mB, %nB : tensor<32x128xi32>
  %limit = tt.splat %lim : i32 -> tensor<32x128xi32>
  %mask = arith.cmpi slt, %diag, %limit : tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %mB : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  %val = tt.load %addr, %mask, %zero : tensor<32x128x!tt.ptr<f16>>
  return %val : tensor<32x128xf16>
}
