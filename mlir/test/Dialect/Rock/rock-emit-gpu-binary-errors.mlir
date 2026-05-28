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

// -----

// Verifies that a keepalive result (tagged `rock.bwd_data_store` in the call's
// res_attrs) with a live use triggers a diagnostic. `expandKernelReturns` in
// ConvToGemm.cpp guarantees that appended keepalive results are use_empty at
// the call site; if any pass downstream of it ever wires a keepalive into real
// consumer code, surface it loudly here instead of silently mis-aliasing.
//
// The host function is stashed in `rock.host_functions` so the parser accepts
// `func.call @<llvm.func>` -- `restoreHostFunctions` parses it with
// verifyAfterParse=false and reassigns the restored ops' locations to the
// owning module, so `expected-error @below` on the module anchors correctly.

// expected-error @below {{'func.call' op keepalive result is not empty}}
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.test_keepalive_live_use" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @keepalive_live_use_caller(%arg0: tensor<144xf32>, %arg1: tensor<72xf32>, %arg2: tensor<512xf32>) -> tensor<512xf32> {\n  %0:4 = func.call @test_keepalive_live_use(%arg0, %arg1, %arg2) {res_attrs = [{}, {rock.bwd_data_store}, {rock.bwd_data_store}, {rock.bwd_data_store}]} : (tensor<144xf32>, tensor<72xf32>, tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>)\n  return %0#1 : tensor<512xf32>\n}"
    ]
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_keepalive_live_use(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a live (non-keepalive) call result with no trailing
// tensor/memref operand to alias to triggers a diagnostic. The kernel takes
// one ptr argument so the call has one trailing tensor operand, but produces
// two live tensor results -- result #1 has no operand to be aliased against,
// violating the trailing-operands-as-outputs contract.

// expected-error @below {{'func.call' op live kernel call result has no trailing tensor/memref operand to alias to}}
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.test_contract_violation" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @contract_violation_caller(%arg0: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {\n  %0:2 = func.call @test_contract_violation(%arg0) : (tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>)\n  return %0#0, %0#1 : tensor<1024xf32>, tensor<1024xf32>\n}"
    ]
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_contract_violation(%arg0: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

