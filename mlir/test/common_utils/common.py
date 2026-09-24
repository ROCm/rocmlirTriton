# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import os
import subprocess

from hip import hip


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def get_agents():
    """Return each visible device's `gcnArchName` in HIP device order.

    HIP applies HIP_VISIBLE_DEVICES itself, so index 0 is the device the tests
    will run on. Keep the duplicates and the ordering: callers need to tell a
    homogeneous machine from a mixed one, and picking an arch out of a set
    would vary between runs under hash randomization.
    """
    agents = []
    device_count = hip_check(hip.hipGetDeviceCount())
    for device in range(device_count):
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, device))
        agents.append(props.gcnArchName.decode('utf-8'))

    return agents


def apply_arch_features(config, lit_config):
    """Populate `config.arch`, `config.no_AMD_GPU`, `config.mixed_arch_detected`,
    and the `arch_support_*` booleans from the `amd_arch_db` pybind11 binding.
    Shared by all lit.site.cfg.py.in files so per-arch gating stays in one place.

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
    config.mixed_arch_detected = False
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

    if not agents:
        config.no_AMD_GPU = True
        return

    # Everything downstream -- the %arch substitution, rocmlir-gen --arch, the
    # feature gating below -- takes a single architecture, so describe device 0
    # only. Joining the architectures of a mixed machine would just hand
    # rocmlir-gen an unparseable chipset.
    config.arch = agents[0]
    distinct = sorted(set(agents))
    config.mixed_arch_detected = len(distinct) > 1
    if config.mixed_arch_detected:
        lit_config.note("Visible GPUs have mixed architectures (%s); tests will run on %s. "
                        "Set HIP_VISIBLE_DEVICES to select a different device." %
                        (', '.join(distinct), config.arch))

    chip = config.arch.split(':')[0]
    config.arch_support_accel_fp8 = amd_arch_db.arch_supports_accel_fp8(chip)
    config.arch_support_scaled_gemm = amd_arch_db.arch_supports_scaled_gemm(chip)
    config.arch_support_non_k_packed_scaled_input = (
        amd_arch_db.arch_supports_non_k_packed_scaled_input(chip))
    config.arch_support_kpack = amd_arch_db.get_max_kpack(chip) > 1
    config.arch_prefers_bf16x3_for_f32_dot = (amd_arch_db.prefer_bf16x3_for_f32_dot(chip))


# HIP honours all of these when it enumerates devices, so each one that is set
# already shaped the device list `get_agents` saw.
DEVICE_SELECTION_VARS = ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')


def apply_device_environment(config):
    """Point the test environment at the same device `apply_arch_features` used
    to compute `config.arch`. Call from each lit.cfg.py alongside the other
    environment setup.

    lit scrubs the device-selection variables, so without this a user who
    selects a device gets an arch describing their choice but kernels running
    on device 0. Propagate every variable that is set rather than just the one
    we would have picked: a selection that happens to leave a single
    architecture visible looks homogeneous here, so the '0' fallback below
    would not fire and the mismatch would go unannounced.
    """
    selected = False
    for var in DEVICE_SELECTION_VARS:
        value = os.environ.get(var)
        if value is not None:
            config.environment[var] = value
            selected = True

    if not selected and config.mixed_arch_detected:
        config.environment['HIP_VISIBLE_DEVICES'] = '0'
