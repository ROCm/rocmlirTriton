import os
import shutil
import subprocess
import sys

# hip-python is the in-process GPU enumeration path on Linux; the Windows HIP
# SDK has no published wheel, so tolerate its absence and fall back below.
try:
    from hip import hip
except ImportError:
    hip = None


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def _find_amdgpu_arch(rocm_path=None):
    # amdgpu-arch prints the gfx arch of each installed GPU, one per line. Its
    # location differs by ROCm distribution: <root>/bin (HIP SDK) and
    # <root>/lib/llvm/bin (TheRock). Search the ROCm/HIP root first so we do not
    # pick up an unrelated amdgpu-arch on PATH (e.g. the one bundled with Visual
    # Studio's LLVM, which cannot enumerate AMD GPUs); PATH is the last resort.
    exe = 'amdgpu-arch.exe' if sys.platform == 'win32' else 'amdgpu-arch'
    root = rocm_path or os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH')
    if root:
        for sub in ('bin', os.path.join('lib', 'llvm', 'bin')):
            cand = os.path.join(root, sub, exe)
            if os.path.isfile(cand):
                return cand
    return shutil.which('amdgpu-arch')


def get_agents(rocm_path=None):
    if sys.platform == 'win32':
        env_arch = os.environ.get('ROCMLIR_TEST_TARGET_ARCH')
        if env_arch:
            agents = set(a.strip() for a in env_arch.split(',') if a.strip())
            if len(agents) != 1:
                raise ValueError("ROCMLIR_TEST_TARGET_ARCH requires one architecture")
            return agents

    # Linux primary path: hip-python FFI.
    if hip is not None:
        agents = set()
        device_count = hip_check(hip.hipGetDeviceCount())
        for device in range(device_count):
            props = hip.hipDeviceProp_t()
            hip_check(hip.hipGetDeviceProperties(props, device))
            agents.add(props.gcnArchName.decode('utf-8'))
        return agents

    # Windows fallback: amdgpu-arch (HIP SDK <root>/bin, TheRock
    # <root>/lib/llvm/bin).
    if sys.platform == 'win32':
        tool = _find_amdgpu_arch(rocm_path)
        if tool:
            try:
                out = subprocess.check_output([tool], stderr=subprocess.DEVNULL).decode()
            except (subprocess.CalledProcessError, OSError):
                out = ''
            # Keep only gfx* lines: TheRock's amdgpu-arch prints a
            # "HIP Library Path: ..." banner on stdout before the arch.
            agents = {ln.strip() for ln in out.splitlines() if ln.strip().startswith('gfx')}
            if agents:
                if len(agents) != 1:
                    raise ValueError("amdgpu-arch reported multiple architectures")
                return agents

    raise subprocess.CalledProcessError(1,
                                        'hip-python/amdgpu-arch',
                                        output=b'',
                                        stderr=b'no GPU enumeration mechanism available')


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
        if not hasattr(amd_arch_db, 'is_fast_atomic_add_supported'):
            raise ImportError("amd_arch_db loaded without expected symbols")
    except ImportError as e:
        lit_config.fatal("amd_arch_db pybind11 module not importable (%s); rebuild "
                         "`rocmlir-common-python-test-utils`." % e)

    config.no_AMD_GPU = False
    config.arch = ""
    config.arch_support_atomic_add_f32 = False
    config.arch_support_atomic_add_f16 = False
    config.arch_support_atomic_add_bf16 = False
    config.arch_support_atomic_max_f32 = False
    config.arch_support_accel_fp8 = False
    config.arch_support_scaled_gemm = False
    config.arch_support_non_k_packed_scaled_input = False
    config.arch_support_kpack = False

    if not config.rocm_path:
        return

    try:
        agents = get_agents(config.rocm_path)
    except ValueError as e:
        lit_config.fatal(str(e))
    except (subprocess.CalledProcessError, RuntimeError):
        # RuntimeError: hip-python loaded but a HIP API call failed at runtime.
        config.no_AMD_GPU = True
        return

    if not agents:
        config.no_AMD_GPU = True
        return

    config.arch = ','.join(agents)
    chip = next(iter(agents)).split(':')[0]
    config.arch_support_atomic_add_f32 = amd_arch_db.is_fast_atomic_add_supported(
        chip, amd_arch_db.Dtype.F32)
    config.arch_support_atomic_add_f16 = amd_arch_db.is_fast_atomic_add_supported(
        chip, amd_arch_db.Dtype.F16)
    config.arch_support_atomic_add_bf16 = amd_arch_db.is_fast_atomic_add_supported(
        chip, amd_arch_db.Dtype.BF16)
    config.arch_support_atomic_max_f32 = amd_arch_db.is_fast_atomic_max_supported(
        chip, amd_arch_db.Dtype.F32)
    config.arch_support_accel_fp8 = amd_arch_db.arch_supports_accel_fp8(chip)
    config.arch_support_scaled_gemm = amd_arch_db.arch_supports_scaled_gemm(chip)
    config.arch_support_non_k_packed_scaled_input = (
        amd_arch_db.arch_supports_non_k_packed_scaled_input(chip))
    config.arch_support_kpack = amd_arch_db.get_max_kpack(chip) > 1
