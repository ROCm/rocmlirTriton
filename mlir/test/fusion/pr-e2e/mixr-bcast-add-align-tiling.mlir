// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-driver -c --arch gfx942 -mlir-print-ir-after=rock-lower-stores -o /dev/null 2>&1 | FileCheck %s

module {
    // CHECK-COUNT-2: rock.blockwise_load
    // CHECK: rock.blockwise_load
    // CHECK: arith.addf
    // CHECK: arith.maximumf
    // CHECK: rock.blockwise_store
    // CHECK-NOT: memref.copy

    func.func @test(%arg0: !migraphx.shaped<64xf32, 1>, %arg1: !migraphx.shaped<1x3x224x224xf32, 150528x50176x224x1>, %arg2: !migraphx.shaped<64x3x7x7xf32, 147x49x7x1>) -> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1> attributes{rock.kernel, rock.arch = "gfx942"} {
        %0 = migraphx.broadcast %arg0 {axis = 1:i64, out_lens= [1:i64, 64:i64, 112:i64, 112:i64] } : <64xf32, 1> -> <1x64x112x112xf32, 0x1x0x0>
        %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : <1x3x224x224xf32, 150528x50176x224x1>, <64x3x7x7xf32, 147x49x7x1> -> <1x64x112x112xf32, 802816x12544x112x1>
        %2 = migraphx.add %1, %0 : <1x64x112x112xf32, 802816x12544x112x1>, <1x64x112x112xf32, 0x1x0x0> -> <1x64x112x112xf32, 802816x12544x112x1>
        %3 = migraphx.relu %2 : <1x64x112x112xf32, 802816x12544x112x1> -> <1x64x112x112xf32, 802816x12544x112x1>
        return %3 : !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>
    }
}
