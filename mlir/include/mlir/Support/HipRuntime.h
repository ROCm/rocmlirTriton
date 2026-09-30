//===- HipRuntime.h - Include HIP's host headers portably -------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Include this instead of <hip/hip_runtime.h>. On a Clang-based compiler that
// is all it does. Under cl.exe it applies the workarounds HIP's headers need
// first, so a caller never has to order two includes correctly: there is no
// window in which a <hip/...> header could be seen unprepared.
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
//  2. HIP's own non-Clang fallback for vector types is broken.
//     amd_hip_vector_types.h:39-45 reads:
//         #if defined(__has_attribute)
//         #if __has_attribute(ext_vector_type)
//         #define __NATIVE_VECTOR__(n, T) T __attribute__((ext_vector_type(n)))
//         #else
//         #define __NATIVE_VECTOR__(n, T) alignas(n * sizeof(T)) T[n]
//         #endif
//
//     MSVC has no __has_attribute, but LLVM's Support/Compiler.h supplies one:
//         #ifndef __has_attribute
//         # define __has_attribute(x) 0
//         #endif
//     so any translation unit that includes an LLVM header before a HIP header
//     takes the `#else` branch, which expands line 92 to
//         using Native_vec_ = alignas(1 * sizeof(T)) T[1];
//     and `alignas` is not permitted in a type-id. The result is C2059
//     "syntax error: 'attribute specifier'" followed by cascading C2504s.
//
//     Undefining __has_attribute is what avoids this: HIP then takes the outer
//     `#else` at line 819, which declares the vector types without consulting
//     __NATIVE_VECTOR__ at all, so that macro needs no definition from us.
//
// Each macro is pushed before it is touched and popped immediately after the
// HIP include, so none of it is visible to the rest of the translation unit.
// That matters most for __has_attribute: leaving it undefined would turn any
// later unguarded `#if __has_attribute(x)` into a hard preprocessor error
// rather than letting it evaluate to 0 as LLVM intends.
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

#ifndef MLIR_SUPPORT_HIPRUNTIME_H
#define MLIR_SUPPORT_HIPRUNTIME_H

#if defined(_MSC_VER) && !defined(__clang__)

// (1) Neutralise GCC/Clang attribute syntax, e.g. amd_hip_vector_types.h:58.
#pragma push_macro("__attribute__")
#undef __attribute__
#define __attribute__(x)

// (2) Steer HIP away from its broken non-Clang vector fallback, as above.
#pragma push_macro("__has_attribute")
#undef __has_attribute

#endif // _MSC_VER && !__clang__

#include <hip/hip_runtime.h>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma pop_macro("__has_attribute")
#pragma pop_macro("__attribute__")
#endif

#endif // MLIR_SUPPORT_HIPRUNTIME_H
