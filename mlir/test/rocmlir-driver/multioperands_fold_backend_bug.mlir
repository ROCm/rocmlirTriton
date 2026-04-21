// This test is just to ensure the backend LLVM compiler does not crash while compiling this kernel. It was crashing earlier and fixed by https://github.com/llvm/llvm-project/pull/148205.
// RUN: sed -e 's/##TOKEN_ARCH##/%arch/g' %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel --arch %arch | rocmlir-driver -c --arch %arch | FileCheck %s
// CHECK: triton.hsaco

module {
  func.func @test(%arg0: !migraphx.shaped<1x40x9419x128xf16, 48225280x1205632x128x1>, %arg1: !migraphx.shaped<1x9419x40x128xf16, 48225280x5120x128x1>) -> !migraphx.shaped<1x40x9419x9419xf16, 3548702440x88717561x9419x1> attributes {rock.kernel = "mixr", rock.arch = "##TOKEN_ARCH##"} {
    %0 = migraphx.literal(dense<8.831780e-02> : tensor<1xf16>) : <1xf16, 1>
    %1 = migraphx.transpose %arg1 {permutation = [0, 2, 3, 1]} : <1x9419x40x128xf16, 48225280x5120x128x1> -> <1x40x128x9419xf16, 48225280x128x1x5120>
    %2 = migraphx.dot %arg0, %1 {perf_config = "gemm:v1:128,256,32,1,1,4,16,1,2,0,0"} : <1x40x9419x128xf16, 48225280x1205632x128x1>, <1x40x128x9419xf16, 48225280x128x1x5120> -> <1x40x9419x9419xf16, 3548702440x88717561x9419x1>
    %3 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 40, 9419, 9419]} : <1xf16, 1> -> <1x40x9419x9419xf16, 0x0x0x0>
    %4 = migraphx.mul %2, %3 : <1x40x9419x9419xf16, 3548702440x88717561x9419x1>, <1x40x9419x9419xf16, 0x0x0x0> -> <1x40x9419x9419xf16, 3548702440x88717561x9419x1>
    return %4 : !migraphx.shaped<1x40x9419x9419xf16, 3548702440x88717561x9419x1>
  }
}
 
