// The extra rocmlir-opt calls check IR validity

// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm -g 1 -m 1 -k 1 -n 1 | rocmlir-opt --mlir-print-local-scope | FileCheck %s '-D$ITYPE=f32' '-D$OTYPE=f32' --check-prefixes=ALLUNIT
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm -g 2 -m 1 -k 1 -n 1 | rocmlir-opt --mlir-print-local-scope | FileCheck %s '-D$ITYPE=f32' '-D$OTYPE=f32' --check-prefixes=ONLYG
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm -g 1 -m 2 -k 1 -n 1 | rocmlir-opt --mlir-print-local-scope | FileCheck %s '-D$ITYPE=f32' '-D$OTYPE=f32' --check-prefixes=ONLYM
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm -g 1 -m 1 -k 2 -n 1 | rocmlir-opt --mlir-print-local-scope | FileCheck %s '-D$ITYPE=f32' '-D$OTYPE=f32' --check-prefixes=ONLYK
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm -g 1 -m 1 -k 1 -n 2 | rocmlir-opt --mlir-print-local-scope | FileCheck %s '-D$ITYPE=f32' '-D$OTYPE=f32' --check-prefixes=ONLYN

// ALLUNIT-LABEL: module
// ALLUNIT-NEXT: func.func @rock_gemm
// ALLUNIT-SAME: ([[arg0:%.+]]: tensor<1x[[$ITYPE]]>, [[arg1:%.+]]: tensor<1x[[$ITYPE]]>, [[arg2:%.+]]: tensor<1x[[$OTYPE]]>)
// ALLUNIT-SAME: attributes {rock.arch = "{{.*}}", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = {{.*}}, rock.num_cu = {{.*}}}
// ALLUNIT-NEXT: [[gemmA:%.+]] = rock.transform [[arg0]]
// ALLUNIT-SAME: <Unmerge{1} ["k"]
// ALLUNIT-SAME: <AddDim{1} ["g"]
// ALLUNIT-SAME: <AddDim{1} ["m"]
// ALLUNIT-NEXT: [[gemmB:%.+]] = rock.transform [[arg1]]
// ALLUNIT-SAME: <Unmerge{1} ["n"]
// ALLUNIT-SAME: <AddDim{1} ["g"]
// ALLUNIT-SAME: <AddDim{1} ["k"]
// ALLUNIT-NEXT: [[result:%.+]] = rock.gemm [[gemmA]] * [[gemmB]] : tensor<1x1x1x[[$ITYPE]]> * tensor<1x1x1x[[$ITYPE]]> -> tensor<1x1x1x[[$OTYPE]]>
// ALLUNIT-NEXT: [[flat:%.+]] = rock.transform [[result]]
// ALLUNIT-NEXT: rock.store [[flat]] to [[arg2]] by set

// ONLYG-LABEL: module
// ONLYG-NEXT: func.func @rock_gemm
// ONLYG-SAME: ([[arg0:%.+]]: tensor<2x[[$ITYPE]]>, [[arg1:%.+]]: tensor<2x[[$ITYPE]]>, [[arg2:%.+]]: tensor<2x[[$OTYPE]]>)
// ONLYG-SAME: attributes {rock.arch = "{{.*}}", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = {{.*}}, rock.num_cu = {{.*}}}
// ONLYG-NEXT: [[gemmA:%.+]] = rock.transform [[arg0]]
// ONLYG-SAME: <Unmerge{2} ["g"]
// ONLYG-SAME: <AddDim{1} ["m"]
// ONLYG-SAME: <AddDim{1} ["k"]
// ONLYG-NEXT: [[gemmB:%.+]] = rock.transform [[arg1]]
// ONLYG-SAME: <Unmerge{2} ["g"]
// ONLYG-SAME: <AddDim{1} ["k"]
// ONLYG-SAME: <AddDim{1} ["n"]
// ONLYG-NEXT: [[result:%.+]] = rock.gemm [[gemmA]] * [[gemmB]] : tensor<2x1x1x[[$ITYPE]]> * tensor<2x1x1x[[$ITYPE]]> -> tensor<2x1x1x[[$OTYPE]]>
// ONLYG-NEXT: [[flat:%.+]] = rock.transform [[result]]
// ONLYG-NEXT: rock.store [[flat]] to [[arg2]] by set

// ONLYM-LABEL: module
// ONLYM-NEXT: func.func @rock_gemm
// ONLYM-SAME: ([[arg0:%.+]]: tensor<2x[[$ITYPE]]>, [[arg1:%.+]]: tensor<1x[[$ITYPE]]>, [[arg2:%.+]]: tensor<2x[[$OTYPE]]>)
// ONLYM-SAME: attributes {rock.arch = "{{.*}}", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = {{.*}}, rock.num_cu = {{.*}}}
// ONLYM-NEXT: [[gemmA:%.+]] = rock.transform [[arg0]]
// ONLYM-SAME: <Unmerge{2} ["m"]
// ONLYM-SAME: <AddDim{1} ["g"]
// ONLYM-SAME: <AddDim{1} ["k"]
// ONLYM-NEXT: [[gemmB:%.+]] = rock.transform [[arg1]]
// ONLYM-SAME: <Unmerge{1} ["n"]
// ONLYM-SAME: <AddDim{1} ["g"]
// ONLYM-SAME: <AddDim{1} ["k"]
// ONLYM-NEXT: [[result:%.+]] = rock.gemm [[gemmA]] * [[gemmB]] : tensor<1x2x1x[[$ITYPE]]> * tensor<1x1x1x[[$ITYPE]]> -> tensor<1x2x1x[[$OTYPE]]>
// ONLYM-NEXT: [[flat:%.+]] = rock.transform [[result]]
// ONLYM-NEXT: rock.store [[flat]] to [[arg2]] by set

// ONLYK-LABEL: module
// ONLYK-NEXT: func.func @rock_gemm
// ONLYK-SAME: ([[arg0:%.+]]: tensor<2x[[$ITYPE]]>, [[arg1:%.+]]: tensor<2x[[$ITYPE]]>, [[arg2:%.+]]: tensor<1x[[$OTYPE]]>)
// ONLYK-SAME: attributes {rock.arch = "{{.*}}", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = {{.*}}, rock.num_cu = {{.*}}}
// ONLYK-NEXT: [[gemmA:%.+]] = rock.transform [[arg0]]
// ONLYK-SAME: <Unmerge{2} ["k"]
// ONLYK-SAME: <AddDim{1} ["g"]
// ONLYK-SAME: <AddDim{1} ["m"]
// ONLYK-NEXT: [[gemmB:%.+]] = rock.transform [[arg1]]
// ONLYK-SAME: <Unmerge{2} ["k"]
// ONLYK-SAME: <AddDim{1} ["g"]
// ONLYK-SAME: <AddDim{1} ["n"]
// ONLYK-NEXT: [[result:%.+]] = rock.gemm [[gemmA]] * [[gemmB]] : tensor<1x1x2x[[$ITYPE]]> * tensor<1x2x1x[[$ITYPE]]> -> tensor<1x1x1x[[$OTYPE]]>
// ONLYK-NEXT: [[flat:%.+]] = rock.transform [[result]]
// ONLYK-NEXT: rock.store [[flat]] to [[arg2]] by set

// ONLYN-LABEL: module
// ONLYN-NEXT: func.func @rock_gemm
// ONLYN-SAME: ([[arg0:%.+]]: tensor<1x[[$ITYPE]]>, [[arg1:%.+]]: tensor<2x[[$ITYPE]]>, [[arg2:%.+]]: tensor<2x[[$OTYPE]]>)
// ONLYN-SAME: attributes {rock.arch = "{{.*}}", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = {{.*}}, rock.num_cu = {{.*}}}
// ONLYN-NEXT: [[gemmA:%.+]] = rock.transform [[arg0]]
// ONLYN-SAME: <Unmerge{1} ["k"]
// ONLYN-SAME: <AddDim{1} ["g"]
// ONLYN-SAME: <AddDim{1} ["m"]
// ONLYN-NEXT: [[gemmB:%.+]] = rock.transform [[arg1]]
// ONLYN-SAME: <Unmerge{2} ["n"]
// ONLYN-SAME: <AddDim{1} ["g"]
// ONLYN-SAME: <AddDim{1} ["k"]
// ONLYN-NEXT: [[result:%.+]] = rock.gemm [[gemmA]] * [[gemmB]] : tensor<1x1x1x[[$ITYPE]]> * tensor<1x1x2x[[$ITYPE]]> -> tensor<1x1x2x[[$OTYPE]]>
// ONLYN-NEXT: [[flat:%.+]] = rock.transform [[result]]
// ONLYN-NEXT: rock.store [[flat]] to [[arg2]] by set
