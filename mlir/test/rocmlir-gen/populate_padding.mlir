// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --padding_h=0 -batchsize=32 -in_channels=32 -out_channels=256 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1  --padding_w_l=1 --padding_w_r=2 --mlir-print-local-scope | FileCheck %s --check-prefix=Padding_One
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --padding_h=3 -batchsize=32 -in_channels=32 -out_channels=256 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1  --padding_w_l=1 --padding_w_r=2 --mlir-print-local-scope | FileCheck %s --check-prefix=Padding_Two

// Padding_One-LABEL: func.func @rock_conv_gkc01_ngc01_ngk01
// Padding_One-SAME: ([[arg0:%.+]]: tensor<8192xf32>, [[arg1:%.+]]: tensor<200704xf32>, [[arg2:%.+]]: tensor<1949696xf32>)
// Padding_One-SAME: attributes {rock.arch = "{{.*}}", rock.enable_splitk_for_tuning, rock.kernel = 0 : i32, rock.num_chiplets = {{.*}}, rock.num_cu = {{.*}}}
// Padding_One-NEXT: [[exp0:%.+]] = rock.transform [[arg0]] by
// Padding_One-SAME: Unmerge{256, 32}
// Padding_One-SAME: AddDim{1} ["g"]
// Padding_One-SAME: AddDim{1} ["0"]
// Padding_One-SAME: AddDim{1} ["1"]
// Padding_One-NEXT: [[exp1:%.+]] = rock.transform [[arg1]] by
// Padding_One-SAME: Unmerge{32, 32, 14, 14}
// Padding_One-SAME: AddDim{1} ["gi"]
// Padding_One-NEXT: [[conv:%.+]] = rock.conv([[exp0]], [[exp1]]) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 1 : index, 2 : index], strides = [1 : index, 1 : index]} : tensor<1x256x32x1x1xf32>, tensor<32x1x32x14x14xf32> -> tensor<32x1x256x14x17xf32>
// Padding_One-NEXT: [[flat:%.+]] = rock.transform [[conv]] by
// Padding_One: rock.store [[flat]] to [[arg2]] by {{.*}}set

// Padding_Two-LABEL: func.func @rock_conv_gkc01_ngc01_ngk01
// Padding_Two-SAME: ([[arg0:%.+]]: tensor<8192xf32>, [[arg1:%.+]]: tensor<200704xf32>, [[arg2:%.+]]: tensor<2785280xf32>)
// Padding_Two-SAME: attributes {rock.arch = "{{.*}}", rock.enable_splitk_for_tuning, rock.kernel = 0 : i32, rock.num_chiplets = {{.*}}, rock.num_cu = {{.*}}}
// Padding_Two-NEXT: [[exp0:%.+]] = rock.transform [[arg0]] by
// Padding_Two-SAME: Unmerge{256, 32}
// Padding_Two-SAME: AddDim{1} ["g"]
// Padding_Two-SAME: AddDim{1} ["0"]
// Padding_Two-SAME: AddDim{1} ["1"]
// Padding_Two-NEXT: [[exp1:%.+]] = rock.transform [[arg1]] by
// Padding_Two-SAME: Unmerge{32, 32, 14, 14}
// Padding_Two-SAME: AddDim{1} ["gi"]
// Padding_Two-NEXT: [[conv:%.+]] = rock.conv([[exp0]], [[exp1]]) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [3 : index, 3 : index, 1 : index, 2 : index], strides = [1 : index, 1 : index]} : tensor<1x256x32x1x1xf32>, tensor<32x1x32x14x14xf32> -> tensor<32x1x256x20x17xf32>
// Padding_Two-NEXT: [[flat:%.+]] = rock.transform [[conv]] by
// Padding_Two: rock.store [[flat]] to [[arg2]] by {{.*}}set
