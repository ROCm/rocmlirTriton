// RUN: rocmlir-opt %s --rock-prepare-llvm="allow-flush-denorm=true" -split-input-file | FileCheck %s
// RUN: rocmlir-opt %s --rock-prepare-llvm="allow-flush-denorm=false" -split-input-file | FileCheck %s --check-prefix=NOFLUSH

// CHECK-LABEL: @gep_inbounds
// CHECK-SAME: (%[[BASE:.+]]: !llvm.ptr {{.*}}, %[[IDX:.+]]: i64)
llvm.func @gep_inbounds(%base: !llvm.ptr {llvm.noalias}, %idx: i64) -> (!llvm.ptr) attributes {rock.kernel} {
  // CHECK: = llvm.getelementptr inbounds %[[BASE]][%[[IDX]]]
  %p0 = llvm.getelementptr %base[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.return %p0 : !llvm.ptr
}

// -----

// addrspace 7 (buffer fat pointers) must NOT get inbounds
// CHECK-LABEL: @gep_no_inbounds_p7
// CHECK-SAME: (%[[BASE:.+]]: !llvm.ptr<7> {{.*}}, %[[IDX:.+]]: i32)
llvm.func @gep_no_inbounds_p7(%base: !llvm.ptr<7> {llvm.noalias}, %idx: i32) -> (!llvm.ptr<7>) attributes {rock.kernel} {
  // CHECK: = llvm.getelementptr %[[BASE]][%[[IDX]]]
  // CHECK-NOT: inbounds
  %p0 = llvm.getelementptr %base[%idx] : (!llvm.ptr<7>, i32) -> !llvm.ptr<7>, f32
  llvm.return %p0 : !llvm.ptr<7>
}

// -----

// CHECK-DAG: #[[$DOMAIN:.+]] = #llvm.alias_scope_domain<id = distinct[{{[0-9]+}}]<>, description = "alias_scopes">
// CHECK-DAG: #[[$AS0:.+]] = #llvm.alias_scope<id = distinct[{{[0-9]+}}]<>, domain = #[[$DOMAIN]], description = "arg1">
// CHECK-DAG: #[[$AS1:.+]] = #llvm.alias_scope<id = distinct[{{[0-9]+}}]<>, domain = #[[$DOMAIN]], description = "arg2">
llvm.func @alias_scopes(%arg0: f32, %arg1: !llvm.ptr {llvm.noalias}, %arg2: !llvm.ptr {llvm.noalias}) attributes {rock.kernel} {
  // CHECK: llvm.load
  // CHECK-SAME: alias_scopes = [#[[$AS0]]]
  // CHECK-SAME: noalias_scopes = [#[[$AS1]]]
  %v = llvm.load %arg1 : !llvm.ptr -> f32
  %w = llvm.fadd %v, %arg0 : f32
  // CHECK: llvm.store
  // CHECK-SAME: alias_scopes = [#[[$AS1]]]
  // CHECK-SAME: noalias_scopes = [#[[$AS0]]]
  llvm.store %w, %arg2 : f32, !llvm.ptr
  llvm.return
}

// -----

// Buffer ops (rocdl.raw.ptr.buffer.*) also implement AliasAnalysisOpInterface,
// so alias scopes should be attached via make_buffer_rsrc -> arg tracing.
// CHECK-DAG: #[[$BDOMAIN:.+]] = #llvm.alias_scope_domain<id = distinct[{{[0-9]+}}]<>, description = "buffer_alias_scopes">
// CHECK-DAG: #[[$BAS0:.+]] = #llvm.alias_scope<id = distinct[{{[0-9]+}}]<>, domain = #[[$BDOMAIN]], description = "arg0">
// CHECK-DAG: #[[$BAS1:.+]] = #llvm.alias_scope<id = distinct[{{[0-9]+}}]<>, domain = #[[$BDOMAIN]], description = "arg1">
llvm.func @buffer_alias_scopes(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}) attributes {rock.kernel} {
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %extent = llvm.mlir.constant(128 : i64) : i64
  %stride = llvm.mlir.constant(0 : i16) : i16
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %rsrc0 = rocdl.make.buffer.rsrc %arg0, %stride, %extent, %flags : !llvm.ptr to <8>
  %rsrc1 = rocdl.make.buffer.rsrc %arg1, %stride, %extent, %flags : !llvm.ptr to <8>
  // CHECK: rocdl.raw.ptr.buffer.load
  // CHECK-SAME: alias_scopes = [#[[$BAS0]]]
  // CHECK-SAME: noalias_scopes = [#[[$BAS1]]]
  %v = rocdl.raw.ptr.buffer.load %rsrc0, %c0, %c0, %c0 : f32
  // CHECK: rocdl.raw.ptr.buffer.store
  // CHECK-SAME: alias_scopes = [#[[$BAS1]]]
  // CHECK-SAME: noalias_scopes = [#[[$BAS0]]]
  rocdl.raw.ptr.buffer.store %v, %rsrc1, %c0, %c0, %c0 : f32
  llvm.return
}

// -----

// CHECK-LABEL: @invariant_load
llvm.func @invariant_load(%arg0: f32, %arg1: !llvm.ptr {llvm.noalias, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.writeonly}) attributes {rock.kernel} {
  // CHECK: llvm.load
  // CHECK-SAME: invariant
  %v = llvm.load %arg1 : !llvm.ptr -> f32
  %w = llvm.fadd %v, %arg0 : f32
  llvm.store %w, %arg2 : f32, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: @atomic_metadata
llvm.func @atomic_metadata(%arg0: !llvm.ptr<1> {llvm.noalias}, %arg1: i32, %arg2: i32) attributes {rock.kernel} {
  // CHECK: llvm.atomicrmw
  // CHECK-SAME: syncscope("agent-one-as") monotonic
  // CHECK-SAME: rocdl.ignore_denormal_mode
  // CHECK-SAME: rocdl.no_fine_grained_memory
  // CHECK-SAME: rocdl.no_remote_memory
  // NOFLUSH: llvm.atomicrmw
  // NOFLUSH-SAME: syncscope("agent-one-as") monotonic
  // NOFLUSH-NOT: rocdl.ignore_denormal_mode
  // NOFLUSH-SAME: rocdl.no_fine_grained_memory
  // NOFLUSH-SAME: rocdl.no_remote_memory
  %v1 = llvm.atomicrmw add %arg0, %arg1 seq_cst : !llvm.ptr<1>, i32
  // CHECK: llvm.cmpxchg
  // CHECK-SAME: syncscope("agent-one-as") monotonic monotonic
  // CHECK-SAME: rocdl.no_fine_grained_memory
  // CHECK-SAME: rocdl.no_remote_memory
  // CHECK-NOT: rocdl.ignore_denormal_mode
  %v2 = llvm.cmpxchg %arg0, %v1, %arg2 seq_cst seq_cst : !llvm.ptr<1>, i32
  llvm.return
}

// -----

// Non-kernel functions should be skipped entirely
// CHECK-LABEL: @non_kernel_skipped
// CHECK-NOT: amdgpu-no-heap-ptr
llvm.func @non_kernel_skipped(%base: !llvm.ptr, %idx: i64) -> (!llvm.ptr) {
  // CHECK: llvm.getelementptr %{{.*}}[%{{.*}}]
  // CHECK-NOT: inbounds
  %p0 = llvm.getelementptr %base[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.return %p0 : !llvm.ptr
}

// -----

// Every kernel is marked amdgpu-no-heap-ptr, which drops the device-heap
// implicit arg so the runtime never launches the one-time __amd_rocclr_initHeap
// kernel. Rock kernels never call a device allocator, so this is unconditional.
// CHECK-LABEL: @no_heap_ptr
// CHECK-SAME: passthrough = ["amdgpu-no-heap-ptr"]
llvm.func @no_heap_ptr(%arg0: !llvm.ptr {llvm.noalias}) attributes {rock.kernel} {
  llvm.return
}

// -----

// An existing passthrough list is preserved and the heap attr appended.
// CHECK-LABEL: @no_heap_ptr_merge
// CHECK-SAME: passthrough = ["amdgpu-unsafe-fp-atomics", "amdgpu-no-heap-ptr"]
llvm.func @no_heap_ptr_merge() attributes {rock.kernel, passthrough = ["amdgpu-unsafe-fp-atomics"]} {
  llvm.return
}
