# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import subprocess

import amd_arch_db


def get_agents():
    agents = set()
    for device in range(amd_arch_db.get_native_device_count()):
        agent = amd_arch_db.get_native_arch(device)
        if agent is not None:
            agents.add(agent)

    return agents


def apply_arch_features(config, lit_config):
    """Populate `config.arch`, `config.no_AMD_GPU`, and the `arch_support_*`
    booleans from the `amd_arch_db` pybind11 binding. Shared by all
    lit.site.cfg.py.in files so per-arch gating stays in one place.

    Fatals out if the binding isn't importable; the hasattr probe rejects the
    empty namespace-package shadow when the .so is missing but the sibling
    build subdir of the same name is on sys.path.
    """
    try:
        import amd_arch_db
        if not hasattr(amd_arch_db, 'get_isa_family'):
            raise ImportError("amd_arch_db loaded without expected symbols")
    except ImportError as e:
        lit_config.fatal("amd_arch_db pybind11 module not importable (%s); rebuild "
                         "`rocmlir-common-python-test-utils`." % e)

    config.no_AMD_GPU = False
    config.arch = ""
    config.arch_support_accel_fp8 = False
    config.arch_support_scaled_gemm = False
    config.arch_support_non_k_packed_scaled_input = False
    config.arch_support_kpack = False
    config.arch_prefers_bf16x3_for_f32_dot = False

    if not config.rocm_path:
        return

    try:
        agents = get_agents()
    except subprocess.CalledProcessError:
        config.no_AMD_GPU = True
        return

    config.arch = ','.join(agents)
    if not config.arch:
        config.no_AMD_GPU = True
        return

    # Take the first agent for feature gating. Multi-arch CI runners are
    # expected to be homogeneous; if that ever changes, switch this to an
    # all()/any() reduction over agents.
    chip = next(iter(agents)).split(':')[0]
    config.arch_support_accel_fp8 = amd_arch_db.arch_supports_accel_fp8(chip)
    config.arch_support_scaled_gemm = amd_arch_db.arch_supports_scaled_gemm(chip)
    config.arch_support_non_k_packed_scaled_input = (
        amd_arch_db.arch_supports_non_k_packed_scaled_input(chip))
    config.arch_support_kpack = amd_arch_db.get_max_kpack(chip) > 1
    config.arch_prefers_bf16x3_for_f32_dot = (amd_arch_db.prefer_bf16x3_for_f32_dot(chip))
