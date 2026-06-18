// The extra rocmlir-opt calls check IR validity

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,NOTRB,NOTRO
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transO | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,NOTRB,TRO
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transB | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,TRB,NOTRO
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transB -transO | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,TRB,TRO
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,NOTRB,NOTRO
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA -transO | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,NOTRB,TRO
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA -transB | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,TRB,NOTRO
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA -transB -transO | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,TRB,TRO

// NOTRA-DAG: #[[mapAUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 769 + d2)>
// TRA-DAG:   #[[mapAUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 769 + d1) * 1024 + d2)>
// NOTRB-DAG: #[[mapBUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 769 + d1) * 512 + d2)>
// TRB-DAG:   #[[mapBUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 512 + d1) * 769 + d2)>
// NOTRA-DAG: #[[$mapAHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// TRA-DAG:   #[[$mapAHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
// NOTRB-DAG: #[[$mapBHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// TRB-DAG:   #[[$mapBHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// NOTRO-DAG: #[[$mapCHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// TRO-DAG:   #[[$mapCHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
// NOTRA-DAG: #[[$trMapAUnmerge:.*]] = #rock.transform_map<#[[mapAUnmerge]] by [<Unmerge{3, 1024, 769} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 1024, 769] -> [2362368]>
// TRA-DAG:   #[[$trMapAUnmerge:.*]] = #rock.transform_map<#[[mapAUnmerge]] by [<Unmerge{3, 769, 1024} ["g", "k", "m"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 769, 1024] -> [2362368]>
// NOTRB-DAG: #[[$trMapBUnmerge:.*]] = #rock.transform_map<#[[mapBUnmerge]] by [<Unmerge{3, 769, 512} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 769, 512] -> [1181184]>
// TRB-DAG:   #[[$trMapBUnmerge:.*]] = #rock.transform_map<#[[mapBUnmerge]] by [<Unmerge{3, 512, 769} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 512, 769] -> [1181184]>

// CHECK: module attributes {rock.arch = "[[$ARCH:.*]]"}
// CHECK-LABEL: func.func @rock_gemm
// CHECK-SAME: (%[[aRaw:.*]]: tensor<2362368xf32>, %[[bRaw:.*]]: tensor<1181184xf32>, %[[cRaw:.*]]: tensor<1572864xf32>)
// CHECK-SAME: attributes {rock.arch = "[[$ARCH]]", rock.enable_splitk_for_tuning, rock.kernel
// CHECK-NEXT: %[[a:.*]] = rock.transform %[[aRaw]] by #[[$trMapAUnmerge]]
// CHECK-NEXT: %[[b:.*]] = rock.transform %[[bRaw]] by #[[$trMapBUnmerge]]
// CHECK-NEXT: rock.gemm
// NOTRA-SAME: %[[a]] *
// TRA-SAME:   tr %[[a]] *
// NOTRB-SAME: %[[b]]
// TRB-SAME:   tr %[[b]]
// NOTRO-SAME: :
// TRO-SAME:   {oTransposed} :
// CHECK-NEXT: rock.transform
// CHECK-NEXT: rock.store
// CHECK-NEXT: return

// CHECK-LABEL: func.func @host_naive_gemm
// CHECK-SAME: (%[[aRaw:.*]]: tensor<2362368xf32>, %[[bRaw:.*]]: tensor<1181184xf32>, %[[cRaw:.*]]: tensor<1572864xf32>) -> tensor<1572864xf32>
// CHECK-NEXT: %[[a:.*]] = tensor.expand_shape %[[aRaw]] [{{\s*}}[0, 1, 2]]
// NOTRA-SAME: into tensor<3x1024x769xf32>
// TRA-SAME:   into tensor<3x769x1024xf32>
// CHECK-NEXT: %[[b:.*]] = tensor.expand_shape %[[bRaw]] [{{\s*}}[0, 1, 2]]
// NOTRB-SAME: into tensor<3x769x512xf32>
// TRB-SAME:   into tensor<3x512x769xf32>
// CHECK-NEXT: %[[cst:.*]] = arith.constant 0.0{{.*}} : f32
// CHECK-NEXT: %[[empty:.*]] = tensor.empty()
// CHECK-NEXT: %[[zero:.*]] = linalg.fill ins(%[[cst]] : f32) outs(%[[empty]] : {{.*}}) -> {{.*}}
// CHECK-NEXT: %[[gemm:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$mapAHost]], #[[$mapBHost]], #[[$mapCHost]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[a]], %[[b]] : tensor<{{.*}}>, tensor<{{.*}}>) outs(%[[zero]] : tensor<{{.*}}>)
// CHECK-NEXT: (%[[aElem:.*]]: f32, %[[bElem:.*]]: f32, %[[cElem:.*]]: f32)
// CHECK-NEXT: %[[mul:.*]] = arith.mulf %[[aElem]], %[[bElem]]
// CHECK-NEXT: %[[add:.*]] = arith.addf %[[mul]], %[[cElem]]
// CHECK-NEXT: linalg.yield %[[add]]
// CHECK:      %[[result:.*]] = tensor.collapse_shape %[[gemm]] [{{\s*}}[0, 1, 2]]
// CHECK-NEXT: return %[[result]]
