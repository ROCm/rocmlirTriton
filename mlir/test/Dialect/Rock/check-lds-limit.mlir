// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-opt -rock-check-lds-limit="arch=gfx90a" --split-input-file %s -verify-diagnostics \
// RUN:   | FileCheck %s --check-prefix=FITS --implicit-check-not=rock.not_applicable
// RUN: rocmlir-opt -rock-check-lds-limit="arch=gfx90a" --split-input-file %s -verify-diagnostics --mlir-print-ir-after-failure 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NA --implicit-check-not=rock.not_applicable
// RUN: not rocmlir-opt -rock-check-lds-limit %s 2>&1 | FileCheck %s --check-prefix=NOARCH

// NOARCH: error: rock-check-lds-limit requires the arch option

// Verifies that a kernel using exactly the LDS of the arch is accepted and left
// unchanged. gfx90a has 65536 bytes of LDS.
// FITS: module attributes {ttg.shared = 65536 : i32}
module attributes {"ttg.shared" = 65536 : i32} {
}

// -----

// Verifies that one byte more is rejected, and that the module is marked
// `rock.not_applicable` so the tuning driver reports the config as N/A rather
// than as a compilation bug.
// expected-error @+2 {{ttg.shared (65537) exceeds LDS limit (65536) for gfx90a}}
// NA: module attributes {rock.not_applicable, ttg.shared = 65537 : i32}
module attributes {"ttg.shared" = 65537 : i32} {
}

// -----

// Verifies that a missing ttg.shared is reported as a pipeline error, without
// marking the module not applicable.
// expected-error @+1 {{ttg.shared attribute not found on module}}
module {
}
