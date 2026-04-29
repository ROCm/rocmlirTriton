// Regression tests for the Triton plugin loader wired up in
// mlir/include/mlir/InitRocMLIRDialects.h::registerTritonDialects.
//
// These tests guard against a class of bug where the env-var name read by
// rocmlir-opt drifts from the one upstream Triton actually uses.
// These tests run rocmlir-opt --help which is enough to call into
// registerRocMLIRDialects and exercise the plugin loading code.
//
// Mirrors external/triton/test/Plugins/test-plugin.mlir but does not require
// the upstream example plugin shared libraries, which are not built in the
// rocmlirTriton project (TRITON_BUILD_PYTHON_MODULE is OFF).

// 1. Baseline: with the env var unset, rocmlir-opt must register dialects
//    and run --help successfully. Guards against making the var mandatory.
// RUN: env -u TRITON_PLUGIN_PATHS rocmlir-opt --help \
// RUN:   | FileCheck %s --check-prefix=CHECK-HELP

// CHECK-HELP: MLIR+Rock modular optimizer driver

// 2. Bug detector: with the env var set to a path that cannot be loaded,
//    rocmlir-opt must abort with a "Could not load library" error.
//
//    If TRITON_PLUGIN_PATHS is silently ignored (because someone renamed
//    the env var upstream), rocmlir-opt will exit 0
//    and this test will fail.
//
//    The current code path uses report_fatal_error -> abort(), so the
//    process dies via SIGABRT. We use `not --crash` so signal-based
//    failures count as the expected outcome.
// RUN: env TRITON_PLUGIN_PATHS=/rocmlir/nonexistent/plugin.so \
// RUN:   not --crash rocmlir-opt --help 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-BAD-PATH

// CHECK-BAD-PATH: Could not load library '/rocmlir/nonexistent/plugin.so'

// 3. Multi-path parsing: the loader splits on ':' and tries each entry.
//    The first bad path should be reported.
// RUN: env TRITON_PLUGIN_PATHS=/rocmlirTriton/nonexistent/a.so:/rocmlirTriton/nonexistent/b.so \
// RUN:   not --crash rocmlir-opt --help 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MULTI

// CHECK-MULTI: Could not load library '/rocmlirTriton/nonexistent/a.so'
