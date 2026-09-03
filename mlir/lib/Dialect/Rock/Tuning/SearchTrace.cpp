//===- SearchTrace.cpp - JSONL trace of a tuning search -------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SearchTrace.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>

using namespace mlir;
using namespace mlir::rock;

TraceWriter::TraceWriter(StringRef path) {
  std::error_code ec;
  auto file = std::make_unique<llvm::raw_fd_ostream>(
      path, ec, llvm::sys::fs::CD_CreateAlways, llvm::sys::fs::FA_Write,
      llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "warning: not tracing the search: cannot open " << path
                 << ": " << ec.message() << "\n";
    return;
  }
  os = std::move(file);
}

void TraceWriter::write(llvm::json::Object record) {
  if (!os)
    return;
  *os << llvm::json::Value(std::move(record)) << "\n";
  os->flush();
}

SharedTrace mlir::rock::openTrace(StringRef path) {
  if (path.empty())
    return nullptr;
  auto writer = std::make_shared<TraceWriter>(path);
  if (!writer->enabled())
    return nullptr;
  return writer;
}

llvm::json::Value mlir::rock::finiteOrNull(double value) {
  if (!std::isfinite(value))
    return nullptr;
  return value;
}
