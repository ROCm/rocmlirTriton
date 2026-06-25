# Kernel Memory Assumptions

This document describes the memory and pointer assumptions that
rocmlirTriton makes about GPU kernels it generates. These assumptions
are split into two categories:

1. **Internal** -- guaranteed by how rocmlirTriton generates kernels.
2. **External** -- requirements on the runtime/caller (e.g. MIGraphX).

Violating any external assumption is undefined behavior.

---

## 1. Internal Assumptions

These hold because of how the rocmlirTriton pipeline constructs kernels.
They do not require any action from callers.

### 1.1 Memory access patterns (`readonly` / `writeonly` / `readnone`)

Each tensor kernel argument falls into exactly one category:

| Pattern | When | LLVM attribute |
|---|---|---|
| **readonly** | Argument is only loaded from (`rock.blockwise_load`) | `llvm.readonly` |
| **writeonly** | Argument is only stored to via non-atomic `set` | `llvm.writeonly` |
| **read+write** | Argument is stored to via `atomic_add` / `atomic_max` (implicit read) | *(none)* |
| **unused** | Argument is never accessed | `llvm.readnone` |

A non-atomic argument that is both read and written is treated as a
compile-time error. This is enforced in `AnalyzeMemoryUse`.

### 1.2 No aliasing between kernel arguments (`noalias`)

Every tensor argument points to a distinct allocation. The compiler
encodes this as `llvm.noalias` on every pointer argument and reinforces
it with LLVM alias-scope metadata on every load/store (because the
AMDGPU backend discards `noalias` during its kernel argument lowering).

### 1.3 Pointers are not captured, freed, or null

| Attribute | Rationale |
|---|---|
| `llvm.nocapture` | Kernel arguments are never stored to memory or returned. |
| `llvm.nofree` | Kernels never call `free` or any deallocation routine. |
| `llvm.nonnull` | A null tensor pointer is never a valid kernel argument. |
| `llvm.noundef` | All arguments are fully initialized before kernel launch. |

### 1.4 SGPR preloading (`inreg`)

All kernel arguments are marked `inreg`, enabling the AMDGPU backend to
preload them into scalar registers (SGPRs) with newer calling
conventions rather than loading them from a kernel argument buffer.
This is applied at the LLVM IR level during code generation
(`TritonToHsaco`). gfx1250 is excluded (following upstream Triton).

### 1.5 GEP `inbounds`

All `llvm.getelementptr` operations (except those on addrspace-7 buffer
fat pointers) are marked `inbounds`. Our generated pointer arithmetic
never produces out-of-bounds addresses.

### 1.6 Invariant loads

Loads from `readonly` arguments are marked `invariant`. The data behind
a readonly pointer does not change for the duration of the kernel
execution, so the backend can freely reorder or CSE these loads.

### 1.7 Relaxed atomic ordering

All `atomicrmw` and `cmpxchg` operations are set to:

- **`monotonic`** ordering -- we only need freedom from data races, not
  acquire/release semantics.
- **`syncscope("agent-one-as")`** -- synchronization is only required
  among work-items on the same GPU agent, not with the host or other
  devices, and we guarantee single-address-space access.

### 1.8 Dereferenceable size

For statically-shaped tensors, `llvm.dereferenceable` is set to the
exact byte size (`ceil(numElements * bitWidth / 8)`). This tells the
backend the full extent of valid memory behind the pointer.

### 1.9 Triton vectorization hints

| Attribute | Value | Rationale |
|---|---|---|
| `tt.divisibility` | 16 | Pointers are 16-byte aligned (128-bit), enabling maximum-width vector loads/stores. |
| `tt.pointer_range` | 32 | Set when tensor size < 2 GB; tells Triton's `ConvertToBufferOps` pass that the tensor fits in a 32-bit offset range, enabling buffer instructions. |

### 1.10 No device-side dynamic allocation (`amdgpu-no-heap-ptr`)

Rock kernels normally never call device-side `malloc`/`new`, so they never
touch the rocclr device heap. `RockPrepareLLVM` marks such a kernel
`amdgpu-no-heap-ptr` (via the LLVM-dialect `passthrough` attribute), which drops
the `hidden_heap_v1` implicit kernel argument from the ABI. Without that
argument the HIP runtime skips the one-time `__amd_rocclr_initHeap` setup kernel
it would otherwise launch at module load.

The attribute is only added after a per-kernel check (mirroring the AMDGPU
attributor's `funcRetrievesHeapPtr`): if the kernel reaches a device allocator
(`malloc`, `free`, or the `__ockl_dm_*` family) — or makes an indirect call we
cannot see through — the heap pointer is kept and the attribute is omitted, so
the kernel still works correctly.

---

## 2. External Assumptions (Requirements on the Caller)

These must be satisfied by whoever launches the kernel. In practice,
this is MIGraphX or the rocmlirTriton test harness.

### 2.1 Coarse-grained device memory (`no_fine_grained_memory`)

All tensor pointers must point to **coarse-grained device-local memory**
(i.e. `hipMalloc` or equivalent). Fine-grained memory (system memory,
`hipMallocManaged` with fine-grained coherence, or memory allocated with
`hipExtMallocWithFlags(..., hipDeviceMallocFinegrained)`) is **not
supported**.

This assumption is encoded as `rocdl.no_fine_grained_memory` on every
atomic operation. Without it, LLVM cannot emit native hardware atomics
on architectures like RDNA3 (gfx1100) and CDNA2 (gfx90a), falling back
to expensive CAS loops instead.

### 2.2 No remote/peer memory (`no_remote_memory`)

Tensor pointers must reside on the **local device**. Pointers to memory
on a remote GPU (peer memory via `hipDeviceEnablePeerAccess`) are not
supported.

This is encoded as `rocdl.no_remote_memory` on atomic operations.

### 2.3 Denormal flushing on f32 (`allow-flush-denorm`)

We set `denormal-fp-math-f32` to `preserve-sign` on every kernel
function, allowing the hardware to **flush f32 denormals to zero** for
all f32 operations (not just atomics). Additionally, atomic
read-modify-write operations are annotated with
`rocdl.ignore_denormal_mode`, which is needed for native `f32` atomic
add on older architectures (e.g. gfx90a) where the hardware atomic unit
unconditionally flushes f32 denormals regardless of the mode register.
Without this metadata, LLVM would fall back to CAS loops on those
targets. Newer architectures (gfx11+, gfx94x) support denormals in
their atomic units natively, so the metadata is redundant but harmless.

This matches rocMLIR's behavior and is gated by the `allow-flush-denorm`
pipeline option (currently set to `true`).

### 2.4 Pointer alignment (16 bytes)

All tensor pointers must be aligned to at least **16 bytes** (128 bits).
This is the natural alignment of GPU memory allocations from
`hipMalloc` and is encoded as `llvm.align = 16` and `tt.divisibility = 16`.

### 2.5 No overlapping allocations

Each tensor argument must point to a **non-overlapping** memory region.
Two kernel arguments must never alias the same underlying storage
(exception: an atomic argument implicitly reads and writes its own
storage, which is handled correctly). This is assumed when
`AnalyzeMemoryUse` sets `llvm.noalias` on every tensor argument and
when `RockPrepareLLVM` adds per-argument alias scope metadata.

### 2.6 Valid, fully dereferenceable pointers

Tensor pointers must be non-null and the full byte extent of the tensor
must be valid to access. For a `tensor<MxNxf32>`, the pointer must be
valid for at least `M * N * 4` bytes. This is assumed when
`AnalyzeMemoryUse` sets `llvm.nonnull` and `llvm.dereferenceable`.

### 2.7 No concurrent external writes

No external agent (host or another kernel) may **write** to any kernel
argument's memory while the kernel is executing. This applies to both
input and output buffers - concurrent writes to a buffer the kernel
also writes would be a data race. Concurrent **reads** from other
kernels are safe for readonly arguments.
This is implied by `agent-one-as` syncscope on atomics (which only
synchronizes among work-items within the same dispatch) and by
`invariant` on loads from readonly arguments.

### 2.8 Static LDS (no dynamic shared memory)

LDS (shared memory) size is baked into the kernel binary at compile
time. `ResolveKernelLaunchParams` converts Triton's dynamic
`@global_smem` into a statically-sized allocation. The caller must pass
**0** for the dynamic shared memory parameter (`sharedMem` in
`hipExtModuleLaunchKernel` / `hipModuleLaunchKernel`). Passing a
non-zero value would cause the runtime to **add** that amount on top of
the static `.amdhsa_group_segment_fixed_size`, potentially exceeding the
hardware LDS limit and causing a launch failure.

### 2.9 KV-cache attention: dynamic sequence length

Attention kernels with KV-cache support use **statically-shaped** K and V
tensors (compiled to a maximum sequence length `maxSeqLen`) but accept a
per-batch runtime scalar `currentSeqLen` that gives the **last valid key
index** (0-based, inclusive — so `currentSeqLen + 1` tokens are
meaningful).

The kernel shortens its N-loop to `ceil((currentSeqLen + 1) /
NPerBlock)` tiles and masks logits for key positions past
`currentSeqLen` to `-inf` on the last iteration, so they do not affect
the softmax result.

**Caller requirements:**

- K and V buffers must be valid for at least
  `ceil((currentSeqLen + 1) / NPerBlock) * NPerBlock` elements along
  the sequence axis (i.e. `currentSeqLen` rounded up to the next tile
  boundary). Padding values beyond `currentSeqLen` are irrelevant
  (masked to `-inf`), but the memory must be mapped and accessible.
  Note: `llvm.dereferenceable` is set to the full static `maxSeqLen`
  byte size, so strictly speaking the safest option is to allocate the
  full `maxSeqLen` extent; in practice the kernel only accesses the
  tile-rounded region.
- `currentSeqLen` must satisfy `0 <= currentSeqLen <= maxSeqLen - 1`.
  A value >= `maxSeqLen` can drive the N-loop past the static tensor
  extent, causing out-of-bounds loads.
- `currentSeqLen` is a 1-D tensor with one element per batch, matching
  the output batch dimension.

---

## Summary Table

| Assumption | Internal/External | LLVM encoding |
|---|---|---|
| Read/write classification | Internal | `readonly`, `writeonly`, `readnone` |
| No aliasing | Internal | `noalias` + alias scopes |
| No capture/free/null/undef | Internal | `nocapture`, `nofree`, `nonnull`, `noundef` |
| SGPR preloading | Internal | `inreg` |
| GEP in-bounds | Internal | `inbounds` flag |
| Invariant loads | Internal | `invariant` on loads |
| Relaxed atomics | Internal | `monotonic`, `agent-one-as` |
| Dereferenceable extent | Internal | `dereferenceable` |
| Vectorization hints | Internal | `tt.divisibility`, `tt.pointer_range` |
| No device heap (skips initHeap) | Internal | `amdgpu-no-heap-ptr` |
| Coarse-grained memory | **External** | `rocdl.no_fine_grained_memory` |
| Device-local memory | **External** | `rocdl.no_remote_memory` |
| Denormal flushing allowed | **External** | `rocdl.ignore_denormal_mode` |
| 16-byte alignment | **External** | `llvm.align = 16` |
| Non-overlapping regions | **External** | `noalias` |
| Valid pointers | **External** | `nonnull`, `dereferenceable` |
| No concurrent external writes | **External** | `agent-one-as`, `invariant` |
| Static LDS (no dynamic shmem) | **External** | LDS size baked into binary; pass `sharedMem = 0` |
| KV-cache: full allocation required | **External** | Static tensor shape; runtime `currentSeqLen` bounds N-loop |
