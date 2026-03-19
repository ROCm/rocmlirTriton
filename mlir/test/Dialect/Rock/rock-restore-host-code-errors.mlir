// RUN: not rocmlir-opt -rock-restore-host-code="arch=gfx90a" --split-input-file %s 2>&1 | FileCheck %s

// Verifies that a missing rock.grid_size attribute causes a diagnostic
// CHECK: error: 'llvm.func' op missing rock.grid_size.test_no_grid module attribute
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.shared" = 0 : i32,
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.func @test_no_grid(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a malformed prefill_args entry (missing 'index') causes a diagnostic
// CHECK: error: 'llvm.func' op malformed rock.prefill_args.test_bad_prefill: entry missing 'index' IntegerAttr
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.shared" = 0 : i32,
    "rock.grid_size.test_bad_prefill" = 2 : i32,
    "rock.prefill_args.test_bad_prefill" = [
        {value = 0.000000e+00 : f32}
    ],
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.func @test_bad_prefill(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a malformed prefill_args entry (missing 'value') causes a diagnostic
// CHECK: error: 'llvm.func' op malformed rock.prefill_args.test_bad_prefill_no_value: entry missing 'value' attribute
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.shared" = 0 : i32,
    "rock.grid_size.test_bad_prefill_no_value" = 2 : i32,
    "rock.prefill_args.test_bad_prefill_no_value" = [
        {index = 2 : i64}
    ],
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.func @test_bad_prefill_no_value(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}
