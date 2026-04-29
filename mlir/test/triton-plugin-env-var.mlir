// Regression tests for the Triton plugin loader wired up in
// mlir/include/mlir/InitRocMLIRDialects.h::registerTritonDialects.
//
// These tests guard against a class of bug where the env-var name read by
// rocmlir-opt drifts from the one upstream Triton actually uses.
// These tests run rocmlir-opt --help which is enough to call into
// registerRocMLIRDialects and exercise the plugin loading code.
//
// We only need rocmlir-opt --help: registerRocMLIRDialects runs
// unconditionally in main() before any CLI parsing, so --help is enough to
// trigger the plugin-loading code in registerTritonDialects.

// 1. Baseline: with the env var unset, rocmlir-opt must register dialects
//    and run --help successfully. No plugin warning should be printed.
//    Guards against making the var mandatory or printing spurious warnings.
// RUN: env -u TRITON_PLUGIN_PATHS rocmlir-opt --help 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HELP --implicit-check-not="TRITON_EXT_ENABLED"

// CHECK-HELP: MLIR+Rock modular optimizer driver

// 2. Bug detector: with the env var set, the path must show up in
//    upstream's "will not load" warning, proving the value reached
//    loadPlugins(). If TRITON_PLUGIN_PATHS is silently ignored (because
//    someone renamed the env var upstream and the rocMLIR side stops
//    forwarding it, or because the loader call site was deleted),
//    no warning is emitted and this test fails.
// RUN: env TRITON_PLUGIN_PATHS=/rocmlir/nonexistent/plugin.so \
// RUN:   rocmlir-opt --help 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-BAD-PATH

// CHECK-BAD-PATH:      WARNING
// CHECK-BAD-PATH:      not built with TRITON_EXT_ENABLED:
// CHECK-BAD-PATH-NEXT: /rocmlir/nonexistent/plugin.so

// 3. Multi-path parsing: the loader splits on ':' and processes every
//    entry. Each path must produce its own warning, proving the value
//    is parsed rather than treated as a single string.
// RUN: env TRITON_PLUGIN_PATHS=/rocmlirTriton/nonexistent/a.so:/rocmlirTriton/nonexistent/b.so \
// RUN:   rocmlir-opt --help 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MULTI

// CHECK-MULTI:      not built with TRITON_EXT_ENABLED:
// CHECK-MULTI-NEXT: /rocmlirTriton/nonexistent/a.so
// CHECK-MULTI:      not built with TRITON_EXT_ENABLED:
// CHECK-MULTI-NEXT: /rocmlirTriton/nonexistent/b.so
