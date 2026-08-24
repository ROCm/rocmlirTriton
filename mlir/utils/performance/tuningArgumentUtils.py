#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import argparse

from perfCommonUtils import Operation

TUNING_SPACE_CHOICES = ["quick", "full", "exhaustive"]
OPERATION_CHOICES = [operation.name.lower() for operation in Operation]
DATA_TYPE_CHOICES = [
    "f32", "f16", "bf16", "i8", "i8_i32", "i8_i8", "fp8", "fp8_f32", "fp8_fp8", "f4E2M1FN"
]
SCALE_TYPE_CHOICES = ["f32", "f8E8M0FNU"]


def add_common_tuning_arguments(
    parser: argparse.ArgumentParser,
    *,
    target_required: bool = False,
    tuning_space_required: bool = False,
    perf_config_timeout_default: int = 0,
) -> None:
    """Add tuning options shared by tuningRunner and crossCompile."""
    parser.add_argument("--op",
                        "--operation",
                        choices=OPERATION_CHOICES,
                        required=True,
                        help="Operation to tune")
    parser.add_argument("--tuning-space",
                        default=None if tuning_space_required else "full",
                        required=tuning_space_required,
                        choices=TUNING_SPACE_CHOICES,
                        help="Tuning space kind to use")

    target_group = parser.add_argument_group(
        "cross-compilation target identity", None if target_required else
        "Only allowed with --compile-only or --benchmark-artifacts, and required by "
        "--compile-only. Tuning against a live GPU always uses that GPU's identity.")
    target_group.add_argument("--target-arch",
                              required=target_required,
                              default=None,
                              metavar="ARCH",
                              help="Override the target arch instead of discovering it via HIP. "
                              "For cross-compilation, use the full HIP gcnArchName including "
                              "target features (for example, gfx950:sramecc+:xnack-).")
    target_group.add_argument(
        "--target-num-cu",
        required=target_required,
        type=int,
        default=None,
        metavar="N",
        help="Override the target compute-unit count instead of discovering it via rocminfo.")
    target_group.add_argument("--target-num-chiplets",
                              required=target_required,
                              type=int,
                              default=None,
                              metavar="N",
                              help="Override the target chiplet/XCD count.")

    parser.add_argument('--data-type',
                        nargs='+',
                        choices=DATA_TYPE_CHOICES,
                        default=["f32", "f16", "i8"],
                        metavar='TYPE',
                        help="Force a set of data types for gemm tuning. Only used when --op=gemm.")

    parser.add_argument(
        '--scale-type',
        nargs='+',
        choices=SCALE_TYPE_CHOICES,
        default=None,
        metavar='TYPE',
        help="Force a set of scale types for gemm tuning. Only used when --op=gemm.")

    parser.add_argument("--rocmlir-gen-flags",
                        "--rocmlir_gen_flags",
                        type=str,
                        default="",
                        metavar="FLAGS",
                        help="Additional flags to pass to rocmlir-gen")
    parser.add_argument("-d",
                        "--debug",
                        action="store_true",
                        default=False,
                        help="Enable detailed per-iteration measurements")
    parser.add_argument("--debug-quick-tune-data",
                        action="store_true",
                        default=False,
                        help="Enable debug output without detailed measurement arrays")
    parser.add_argument("--verify-winning-config",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Verify the winning perf config against the CPU reference.")
    parser.add_argument("--verify-all-perfconfigs",
                        action="store_true",
                        default=False,
                        help="Verify every successful perf config.")
    parser.add_argument("--timeout",
                        type=int,
                        default=None,
                        metavar="SECONDS",
                        help="Timeout in seconds for each tuning problem")
    parser.add_argument("--num-cpus",
                        type=int,
                        default=None,
                        metavar="N",
                        help="Maximum CPU threads for compilation")
    parser.add_argument("--flush-last-level-cache",
                        action="store_true",
                        default=False,
                        help="Size the cache-flush buffer to the architecture's last-level cache "
                        "(e.g. AMD Infinity Cache) instead of the per-XCD L2 cache size reported "
                        "by the HIP runtime. Defaults to the L2 cache size.")
    parser.add_argument(
        "--perf-config-timeout",
        type=int,
        default=perf_config_timeout_default,
        metavar="SECONDS",
        help="Per-perf-config compilation timeout in seconds (0 = no timeout, compile "
        "in-process). When > 0, each config is compiled in a separate rocmlir-driver "
        "process that is killed if it exceeds this budget; the timed-out config is "
        "skipped (reported as N/A) and tuning continues.")
    parser.add_argument(
        "--allow-commit-mismatch",
        action="store_true",
        default=False,
        help="Allow artifact benchmarking when compile and benchmark commits differ.")
