// Verify rocmlir-driver's --perf-config flag stamps the given perf config onto
// the tunable op before lowering. With no pipeline requested, rocmlir-driver
// only parses, stamps, and reprints the module, so the attribute is observable
// directly in the output.

// Stamp a perf config when the input has none.
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 -n 128 -k 8 -m 256 \
// RUN: | rocmlir-driver --perf-config "gemm:v1:128,64,64,1,1,4,16,1,2,0,0" \
// RUN: | FileCheck %s --check-prefix=SET

// SET: rock.gemm
// SET-SAME: perf_config = "gemm:v1:128,64,64,1,1,4,16,1,2,0,0"

// Override a perf config that is already present on the op.
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 -n 128 -k 8 -m 256 --perf_config "gemm:v1:64,64,16,1,1,4,16,1,1,0,0" \
// RUN: | rocmlir-driver --perf-config "gemm:v1:128,64,64,1,1,4,16,1,2,0,0" \
// RUN: | FileCheck %s --check-prefix=OVERRIDE

// OVERRIDE-NOT: perf_config = "gemm:v1:64,64,16,1,1,4,16,1,1,0,0"
// OVERRIDE: rock.gemm
// OVERRIDE-SAME: perf_config = "gemm:v1:128,64,64,1,1,4,16,1,2,0,0"

// Error out (rather than silently no-op) when --perf-config is given but there
// is no tunable op to stamp. This file parses to an empty module, so there is
// nothing for tuningSetStr to apply the config to.
// RUN: not rocmlir-driver --perf-config "gemm:v1:128,64,64,1,1,4,16,1,2,0,0" %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOOP

// NOOP: Failed to apply --perf-config
// NOOP-SAME: no gemm or gemm+gemm op found
