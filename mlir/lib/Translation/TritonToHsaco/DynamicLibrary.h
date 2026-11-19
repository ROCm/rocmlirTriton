//===- DynamicLibrary.h - Minimal dynamic library loading layer -----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thin portability layer over the native loader APIs: dlopen/dlsym/dlclose on
// Linux and LoadLibrary/GetProcAddress/FreeLibrary on Windows.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TRANSLATION_TRITONTOHSACO_DYNAMICLIBRARY_H
#define MLIR_LIB_TRANSLATION_TRITONTOHSACO_DYNAMICLIBRARY_H

#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mlir {
namespace rock {
namespace dl {

using LibraryHandle = void *;

/// Returns a description of the last loader failure on this thread.
inline std::string lastError() {
#ifdef _WIN32
  DWORD code = GetLastError();
  if (code == 0)
    return "unknown error";
  char *buffer = nullptr;
  DWORD len = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<LPSTR>(&buffer), 0, nullptr);
  std::string message = len && buffer ? std::string(buffer, len)
                                      : "error code " + std::to_string(code);
  if (buffer)
    LocalFree(buffer);
  while (!message.empty() && (message.back() == '\n' || message.back() == '\r' ||
                              message.back() == ' '))
    message.pop_back();
  return message;
#else
  const char *error = dlerror();
  return error ? error : "unknown error";
#endif
}

/// Loads a library by name or path. Returns nullptr on failure; call
/// lastError() for details.
inline LibraryHandle open(const char *name) {
#ifdef _WIN32
  return reinterpret_cast<LibraryHandle>(LoadLibraryA(name));
#else
  return dlopen(name, RTLD_LOCAL | RTLD_NOW);
#endif
}

/// Looks up a symbol. Returns nullptr on failure; call lastError() for details.
inline void *getSymbol(LibraryHandle handle, const char *name) {
  if (!handle)
    return nullptr;
#ifdef _WIN32
  return reinterpret_cast<void *>(
      GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
#else
  return dlsym(handle, name);
#endif
}

/// Unloads a library. Accepts nullptr.
inline void close(LibraryHandle handle) {
  if (!handle)
    return;
#ifdef _WIN32
  FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
  dlclose(handle);
#endif
}

} // namespace dl
} // namespace rock
} // namespace mlir

#endif // MLIR_LIB_TRANSLATION_TRITONTOHSACO_DYNAMICLIBRARY_H
