// RUN: rocmlir-opt --rock-detect-flash-decoding %s | FileCheck %s

// Flash-decoding split-KV promotion in the presence of a KV-cache currentSeqLen.
// When the pass promotes splitKV, the batch dimension of the attention op is
// split; currentSeqLen must be sliced the same way Q/K/V are (fixing the
// split-KV coordinate to 0), otherwise the sliced attention op fails
// batch-size verification. Here batch 24 = 12 heads * 2 split-KV, so after the
// split currentSeqLen goes from tensor<24xi32> to tensor<12xi32>.

#map = affine_map<(d0, d1, d2, d3) -> ((d1 * 256 + d2) * 256 + d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> ((d1 * 256 + d3) * 256 + d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 256 + d2) * 2 + d3) * 128 + d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>
#map6 = affine_map<(d0, d1, d2) -> (0, d0 floordiv 2, d0 mod 2, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 128 + d2)>
#cslmap0 = affine_map<(d0, d1, d2) -> (d1)>
#cslmap1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#cslmap2 = affine_map<(d0) -> (0, d0 floordiv 2, d0 mod 2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{12, 256, 256} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 12, 256, 256] -> [786432]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{12, 256, 256} ["exp1", "exp3", "exp4"] at [1, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [1, 12, 1, 256, 256] -> [786432]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [1, 12, 2, 256, 256] -> [1, 12, 1, 256, 256]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["dim0", "dim1", "dim3", "dim2"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim3", "dim2"] at [0, 1, 3, 2]>] bounds = [1, 12, 256, 256] -> [1, 12, 256, 256]>
#transform_map4 = #rock.transform_map<#map4 by [<Unmerge{12, 256, 2, 128} ["exp1", "exp2", "exp3", "exp4"] at [1, 2, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 12, 256, 2, 128] -> [786432]>
#transform_map5 = #rock.transform_map<#map5 by [<PassThrough ["dim0", "dim1", "dim3", "dim4", "dim2"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim3", "dim4", "dim2"] at [0, 1, 3, 4, 2]>] bounds = [1, 12, 2, 128, 256] -> [1, 12, 256, 2, 128]>
#transform_map6 = #rock.transform_map<#map6 by [<Merge{1, 12, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [24, 256, 256] -> [1, 12, 2, 256, 256]>
#transform_map8 = #rock.transform_map<#map8 by [<Unmerge{24, 256, 128} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [24, 256, 128] -> [786432]>
#transform_map9 = #rock.transform_map<#map6 by [<Merge{1, 12, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [24, 128, 256] -> [1, 12, 2, 128, 256]>
#cslt0 = #rock.transform_map<#cslmap0 by [<Unmerge{12} ["exp1"] at [1] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [1, 12, 1] -> [12]>
#cslt1 = #rock.transform_map<#cslmap1 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [1, 12, 2] -> [1, 12, 1]>
#cslt2 = #rock.transform_map<#cslmap2 by [<Merge{1, 12, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [24] -> [1, 12, 2]>
module {
  // CHECK-LABEL: func.func @mlir_flash_decode_kvcache
  // CHECK: rock.attention{
  // CHECK-NEXT: qk = %{{.*}} * %{{.*}} : tensor<12x256x256xf16>, tensor<12x256x256xf16>
  // The KV-cache currentSeqLen is sliced from tensor<24xi32> to tensor<12xi32>.
  // CHECK-NEXT: currentSeqLen = (%{{.*}} : tensor<12xi32>)
  // CHECK: splitKV = 2 : i32
  func.func @mlir_flash_decode_kvcache(%arg0: tensor<786432xf16>, %arg1: tensor<786432xf16>, %arg2: tensor<786432xf16>, %csl: tensor<12xi32>) -> (tensor<24x256x256xf16>, tensor<24x256xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel = "mixr", rock.num_cu = 48 : i64} {
    %0 = rock.transform %arg1 by #transform_map : tensor<786432xf16> to tensor<1x12x256x256xf16>
    %1 = rock.transform %arg0 by #transform_map1 : tensor<786432xf16> to tensor<1x12x1x256x256xf16>
    %2 = rock.transform %1 by #transform_map2 : tensor<1x12x1x256x256xf16> to tensor<1x12x2x256x256xf16>
    %3 = rock.transform %0 by #transform_map3 : tensor<1x12x256x256xf16> to tensor<1x12x256x256xf16>
    %4 = rock.transform %arg2 by #transform_map4 : tensor<786432xf16> to tensor<1x12x256x2x128xf16>
    %5 = rock.transform %4 by #transform_map5 : tensor<1x12x256x2x128xf16> to tensor<1x12x2x128x256xf16>
    %6 = rock.transform %2 by #transform_map6 : tensor<1x12x2x256x256xf16> to tensor<24x256x256xf16>
    %8 = rock.transform %arg0 by #transform_map8 : tensor<786432xf16> to tensor<24x256x128xf16>
    %9 = rock.transform %5 by #transform_map9 : tensor<1x12x2x128x256xf16> to tensor<24x128x256xf16>
    %c0 = rock.transform %csl by #cslt0 : tensor<12xi32> to tensor<1x12x1xi32>
    %c1 = rock.transform %c0 by #cslt1 : tensor<1x12x1xi32> to tensor<1x12x2xi32>
    %c2 = rock.transform %c1 by #cslt2 : tensor<1x12x2xi32> to tensor<24xi32>
    %result, %lseOut = rock.attention{
     qk = %6 * %8 : tensor<24x256x256xf16>, tensor<24x256x128xf16>
     currentSeqLen = (%c2 : tensor<24xi32>)
     softmax(qk) * %9 : tensor<24x128x256xf16>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, perf_config = "attn:v1:64,64,128,1,1,4,0,1,1,0,0", softmaxType = f32, splitKV = 1 : i32} -> tensor<24x256x256xf16>, tensor<24x256xf32>
    return %result, %lseOut : tensor<24x256x256xf16>, tensor<24x256xf32>
  }
}
