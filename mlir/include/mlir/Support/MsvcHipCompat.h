//===- MsvcHipCompat.h - Let MSVC consume HIP's host headers ----*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Include this instead of <hip/hip_runtime.h>. It applies the workarounds MSVC
// needs and then pulls HIP in itself, so the ordering cannot be got wrong by a
// caller: there is no window in which a <hip/...> header is seen first.
//
// HIP's headers are written for a Clang-based compiler. Two things break under
// cl.exe, and neither is fundamental for host-only use:
//
//  1. GCC/Clang attribute syntax appears unguarded. For example
//     amd_detail/amd_hip_vector_types.h:58 declares
//       template <typename T, unsigned int n>
//       __attribute__((always_inline)) __HOST_DEVICE__ ...
//     and cl.exe cannot parse `__attribute__`, reporting C2988/C2059.
//     These are inlining and alignment hints, not semantics, so defining the
//     macro away is safe for host code.
//
//  2. The native vector layout is already portable: the same header checks
//     `__has_attribute(ext_vector_type)` and falls back to
//     `alignas(n * sizeof(T)) T[n]` when it is unavailable, which is the path
//     cl.exe takes. Nothing to do here.
//
// What this does NOT enable is compiling device code. `__global__` kernels
// still require a Clang-based compiler; this only covers the host runtime API
// (hipModuleLoadData, hipModuleLaunchKernel, events, streams, allocation).
//
// Note also that hip_ext.h's hipExtLaunchKernelGGL calls `pArgs`, which HIP
// only defines inside `#if __HIP_CLANG_ONLY__`. Translation units that include
// hip_ext.h must define `pArgs` themselves first; rocmlir-tuning-driver.cpp
// already does so for the same reason under GCC.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_MSVCHIPCOMPAT_H
#define MLIR_SUPPORT_MSVCHIPCOMPAT_H

#if defined(_MSC_VER) && !defined(__clang__)

// (1) Neutralise GCC/Clang attribute syntax, e.g. amd_hip_vector_types.h:58.
#ifndef __attribute__
#define __attribute__(x)
#endif

// (2) Work around a genuine bug in HIP's own non-Clang fallback.
//
// amd_hip_vector_types.h:39-45 reads:
//     #if defined(__has_attribute)
//     #if __has_attribute(ext_vector_type)
//     #define __NATIVE_VECTOR__(n, T) T __attribute__((ext_vector_type(n)))
//     #else
//     #define __NATIVE_VECTOR__(n, T) alignas(n * sizeof(T)) T[n]
//     #endif
//
// MSVC does not provide __has_attribute, but LLVM's Support/Compiler.h does:
//     #ifndef __has_attribute
//     # define __has_attribute(x) 0
//     #endif
// so any translation unit that includes an LLVM header before a HIP header
// takes the `#else` branch, which expands line 92 to
//     using Native_vec_ = alignas(1 * sizeof(T)) T[1];
// and `alignas` is not permitted in a type-id. The result is C2059
// "syntax error: 'attribute specifier'" followed by cascading C2504s.
//
// A plain array is the correct expansion: the alignment is already supplied by
// the enclosing specialisations, which are declared
// `struct alignas(2 * sizeof(T)) HIP_vector_base<T, 2>` and so on. Defining the
// macro up front also stops HIP from emitting its broken version, since both
// arms of the #if above are guarded.
//
// Both macros must keep HIP's own spelling for the non-Clang fallback. HIP
// guards its definitions with #ifndef, so a definition that differs only in
// formatting would still trip C4005 (macro redefinition), fatal under /WX.
#ifndef __NATIVE_VECTOR__
#define __NATIVE_VECTOR__(n, T) T[n]
#endif
#ifndef __HIP_USE_NATIVE_VECTOR__
#define __HIP_USE_NATIVE_VECTOR__ 0
#endif

// Removing LLVM's `#define __has_attribute(x) 0` fallback is what steers HIP
// into the outer #else above, where neither arm of its broken #if is taken.
//
// Note that this is NOT scoped to the HIP include: the header guard means it
// runs once per translation unit, so __has_attribute stays undefined for
// everything compiled afterwards. Any later header with an unguarded
// `#if __has_attribute(x)` therefore becomes a hard preprocessor error under
// cl.exe instead of quietly evaluating to 0.
#undef __has_attribute

#endif // _MSC_VER && !__clang__

// Deliberately last: every workaround above is in place by the time HIP is
// parsed, for every translation unit, without the caller having to order two
// includes correctly.
#include <hip/hip_runtime.h>

#endif // MLIR_SUPPORT_MSVCHIPCOMPAT_H
