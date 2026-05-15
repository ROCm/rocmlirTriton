// RUN: rocmlir-opt -rock-emit-gpu-binary="arch=gfx90a" -verify-diagnostics --split-input-file %s

// Verifies that a missing rock.grid_size attribute causes a diagnostic
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.shared" = 0 : i32,
    "triton.hsaco" = "DUMMY_HSACO"
} {
  // expected-error @below {{'llvm.func' op missing rock.grid_size.test_no_grid module attribute}}
  llvm.func @test_no_grid(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a malformed prefill_args entry (missing 'index') causes a diagnostic
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.shared" = 0 : i32,
    "rock.grid_size.test_bad_prefill" = 2 : i32,
    "rock.prefill_args.test_bad_prefill" = [
        {value = 0.000000e+00 : f32}
    ],
    "triton.hsaco" = "DUMMY_HSACO"
} {
  // expected-error @below {{'llvm.func' op malformed rock.prefill_args.test_bad_prefill: entry missing 'index' IntegerAttr}}
  llvm.func @test_bad_prefill(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a malformed prefill_args entry (missing 'value') causes a diagnostic
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.shared" = 0 : i32,
    "rock.grid_size.test_bad_prefill_no_value" = 2 : i32,
    "rock.prefill_args.test_bad_prefill_no_value" = [
        {index = 2 : i64}
    ],
    "triton.hsaco" = "DUMMY_HSACO"
} {
  // expected-error @below {{'llvm.func' op malformed rock.prefill_args.test_bad_prefill_no_value: entry missing 'value' attribute}}
  llvm.func @test_bad_prefill_no_value(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a non-dictionary prefill_args entry causes a diagnostic
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.shared" = 0 : i32,
    "rock.grid_size.test_bad_prefill_type" = 2 : i32,
    "rock.prefill_args.test_bad_prefill_type" = [42 : i64],
    "triton.hsaco" = "DUMMY_HSACO"
} {
  // expected-error @below {{'llvm.func' op malformed rock.prefill_args.test_bad_prefill_type: entry is not a DictionaryAttr}}
  llvm.func @test_bad_prefill_type(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}
