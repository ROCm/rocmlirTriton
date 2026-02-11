"""
This script compiles a given config in order to collect the following data
points for each config:
- Blocksize
- Gridsize
- vgpr
- sgpr
- LDS allocated
- Occupancy
- wf_per_wg
- mfma_wmma_instruction

The given config is expected to be a tsv with the following format:
|# arch| numCUs | testVector | perfConfig (exhaustive) |

Usage:
    python3 compileAndCollectTuningData.py --op <operation> <config.tsv>
"""

import argparse
import csv
import os
import re
import sys

from datetime import datetime
from testing_metrics import calculate_gemm_occupancy, calculate_attention_occupancy

# This script expects that ninja ci-performance-scripts has already been run
# so that we have access to perfRunner and perfCommonUtils
import perfRunner
from perfCommonUtils import Operation

# Try to import amd_arch_db; provide actionable error if missing.
try:
    import amd_arch_db
except ModuleNotFoundError as e:
    print(
        "ERROR: Could not import amd_arch_db (pybind11 GPU arch database).\n"
        f"Reason: {e}\n\n"
        "To build it:\n"
        "  1) Manually build amd_arch_db:\n"
        "     ninja amd_arch_db\n"
        "  2) Add the amd_arch_db to PYTHONPATH\n"
    )
    amd_arch_db = None

# Constants for the new result field names
FIELD_NS = 'ns'
FIELD_BLOCKSIZE = 'blocksize'
FIELD_GRIDSIZE = 'gridsize'
FIELD_VGPR_COUNT = 'vgpr_count'
FIELD_VGPR_SPILLS = 'vgpr_spills'
FIELD_SGPR_COUNT = 'sgpr_count'
FIELD_SGPR_SPILLS = 'sgpr_spills'
FIELD_LDS_ALLOCATED = 'lds_allocated'
FIELD_OCCUPANCY = 'occupancy'
FIELD_WF_PER_WG = 'wf_per_wg'
FIELD_MFMA_WMMA_INSTRUCTION = 'mfma_wmma_instruction'
FIELD_PERFCONFIG_ROCMLIR = 'rocmlir_perfconfig'

# Define new data fieldnames (the tuning metrics we're adding)
NEW_DATA_FIELDNAMES = [
    FIELD_PERFCONFIG_ROCMLIR,
    FIELD_NS,
    FIELD_BLOCKSIZE,
    FIELD_GRIDSIZE,
    FIELD_VGPR_COUNT,
    FIELD_VGPR_SPILLS,
    FIELD_SGPR_COUNT,
    FIELD_SGPR_SPILLS,
    FIELD_LDS_ALLOCATED,
    FIELD_OCCUPANCY,
    FIELD_WF_PER_WG,
    FIELD_MFMA_WMMA_INSTRUCTION
]


class TuningData:
    """Class to represent tuning data results."""

    def __init__(self):
        self.ns = None
        self.blocksize = None
        self.gridsize = None
        self.vgpr_count = None
        self.vgpr_spills = None
        self.sgpr_count = None
        self.sgpr_spills = None
        self.lds_allocated = None
        self.occupancy = None
        self.wf_per_wg = None
        self.mfma_wmma_instruction = None
        self.rocmlir_perfconfig = None

    def to_dict(self):
        """Convert to dictionary format for tsv writing."""
        return self.__dict__


def get_perf_config(operation, test_vector, arch, num_cu, num_chiplets):
    """
    Get the performance configuration for the given test vector, architecture,
    number of compute units, and number of chiplets.

    Args:
        operation: The operation type.
        test_vector: The test vector string.
        arch: The architecture string.
        num_cu: The number of compute units.
        num_chiplets: The number of chiplets.

    Returns:
        PerfConfiguration: The performance configuration instance.
    """
    conf_class = perfRunner.PerfConfiguration
    if operation == Operation.ATTENTION:
        conf_class = perfRunner.AttentionConfiguration.from_command_line(
            test_vector.split(sep=' '), arch, num_cu, num_chiplets)
    elif operation == Operation.GEMM:
        conf_class = perfRunner.GemmConfiguration.from_command_line(
            test_vector.split(sep=' '), arch, num_cu, num_chiplets)
    elif operation == Operation.CONV:
        conf_class = perfRunner.ConvConfiguration.from_command_line(
            test_vector.split(sep=' '), arch, num_cu, num_chiplets)
    elif operation == Operation.GEMM_GEMM:
        conf_class = perfRunner.GemmGemmConfiguration.from_command_line(
            test_vector.split(sep=' '), arch, num_cu, num_chiplets)

    return conf_class


def compile_config(conf_class, paths, timestamp):
    rocmlir_gen_options = conf_class.generate_mlir_driver_commandline(
        "", kernel_repeats=None)

    # Build the rocmlir-gen command
    rocmlir_gen_cmd = [paths.mlir_paths.rocmlir_gen_path] + \
        rocmlir_gen_options.split()

    # Build the rocmlir-driver command
    rocmlir_driver_cmd = [
        paths.mlir_paths.rocmlir_driver_path,
        "-c",
        f"--arch={conf_class.arch}",
        "--debug-only=rock-gridwise-gemm-to-blockwise,triton-to-hsaco",
    ]

    # Run rocmlir-gen | rocmlir-driver pipeline
    commands = [rocmlir_gen_cmd, rocmlir_driver_cmd]
    outs, errs = perfRunner.run_pipeline(commands)
    return outs, errs


def parse_mfma_wmma_instructions(content):
    """
    Parse MFMA and WMMA instructions from the debug output.

    Args:
        content: String content of the debug output file

    Returns:
        list: Unique list of MFMA/WMMA instruction names
    """
    # Pattern to match MFMA and WMMA instructions
    full_pattern = r'\b(v_(?:mfma|wmma)_[a-zA-Z0-9_]+)\b'
    full_matches = re.findall(full_pattern, content, re.IGNORECASE)

    # Remove duplicates and sort for consistent output
    unique_instructions = list(set(full_matches))

    # Assert that there is only one unique instruction
    size = len(unique_instructions)
    assert size <= 1, \
           f"Expected exactly one unique MFMA/WMMA instruction, found: {size}"

    return unique_instructions


def compute_wave_distribution(num_waves, m_per_block, n_per_block,
                              matrix_instr_nonkdim):
    """
    Compute how waves are distributed across M and N dimensions.

    This is a port of the warpsPerTile() function from Triton's
    AccelerateAMDMatmul.cpp.

    Args:
        num_waves: Total number of waves in the workgroup.
        m_per_block: M tile size per block.
        n_per_block: N tile size per block.
        matrix_instr_nonkdim: The MFMA/WMMA instruction dimension (e.g. 16).

    Returns:
        tuple: (mPerWave, nPerWave)
    """
    # matrixInstrNonkdim=0 means WMMA mode, the actual WMMA tile is 16x16
    if matrix_instr_nonkdim == 0:
        matrix_instr_nonkdim = 16
    shape_per_warp = (matrix_instr_nonkdim, matrix_instr_nonkdim)
    m_warps = 1
    n_warps = 1

    while m_warps * n_warps < num_waves:
        # Distribute to the dimension with more remaining tiles.
        # The comparison uses (shapePerWarp * 2) for M to bias towards
        # distributing along N first when tiles are equal.
        m_tiles_remaining = m_per_block // (shape_per_warp[0] * 2) // m_warps
        n_tiles_remaining = n_per_block // shape_per_warp[1] // n_warps
        if m_tiles_remaining >= n_tiles_remaining:
            if m_warps < m_per_block // shape_per_warp[0]:
                m_warps *= 2
            else:
                n_warps *= 2
        else:
            n_warps *= 2

    # If N warps exceed the number of N tiles, swap M and N
    if n_warps * shape_per_warp[1] > n_per_block:
        m_warps, n_warps = n_warps, m_warps

    m_per_wave = m_per_block // m_warps
    n_per_wave = n_per_block // n_warps
    return m_per_wave, n_per_wave


def _convert_gemm_to_rocmlir(params):
    """Convert gemm:v1 params to rocMLIR AccelGemm v4 perfConfig string.

    rocmlirTriton gemm:v1 input fields:
        mPerBlock, nPerBlock, kPerBlock, kpack, numCTAs, numWaves,
        matrixInstrNonkdim, splitKFactor, numStages, wavesPerEU,
        gridGroupSize

    rocMLIR AccelGemm v4 output fields and how each is derived:
        mPerBlock        = triton.mPerBlock             (direct)
        nPerBlock        = triton.nPerBlock             (direct)
        kpackPerBlock    = triton.kPerBlock             (since kpack=1)
        mPerWave         = mPerBlock / mWaves           (warpsPerTile algo)
        nPerWave         = nPerBlock / nWaves           (warpsPerTile algo)
        mnPerXdl         = triton.matrixInstrNonkdim    (direct)
        kpack            = 1                            (always 1)
        splitKFactor     = triton.splitKFactor          (direct)
        scheduleVersion  = 1 if numStages<=1 else 2     (Default/DoubleBuffer)
        outputSwizzle    = 2                            (default to let rocMLIR choose)
        wavesPerEU       = triton.wavesPerEU            (direct)
        gridGroupSize    = triton.gridGroupSize         (direct)
        forceUnroll      = 1                            (default, Triton has no equivalent tuning param)
        threadCopyMore   = 1                            (legacy, always 1)
    """
    if len(params) != 11:
        return None

    m_per_block = params[0]
    n_per_block = params[1]
    k_per_block = params[2]
    # kpack = params[3]  # rocmlirTriton kpack (not used in mapping)
    # num_ctas = params[4]  # always 1
    num_waves = params[5]
    matrix_instr_nonkdim = params[6]
    split_k_factor = params[7]
    num_stages = params[8]
    waves_per_eu = params[9]
    grid_group_size = params[10]

    kpack_per_block = k_per_block  # since rocmlirTriton kpack = 1
    m_per_wave, n_per_wave = compute_wave_distribution(
        num_waves, m_per_block, n_per_block, matrix_instr_nonkdim)
    schedule_version = 1 if num_stages <= 1 else 2

    # AccelGemm v4 format (see RockAttrDefs.td):
    # v4:mPerBlock,nPerBlock,kpackPerBlock,mPerWave,nPerWave,mnPerXdl,
    #    kpack,splitKFactor,scheduleVersion,outputSwizzle,wavesPerEU,
    #    gridGroupSize,forceUnroll,threadCopyMore
    return (f"v4:{m_per_block},{n_per_block},{kpack_per_block},"
            f"{m_per_wave},{n_per_wave},{matrix_instr_nonkdim},"
            f"1,{split_k_factor},{schedule_version},"
            f"2,{waves_per_eu},{grid_group_size},"
            f"1,1")


def _convert_attn_to_rocmlir(params):
    """Convert attn:v1 params to rocMLIR GemmGemm v3 (attn:v3) perfConfig string.

    rocmlirTriton attn:v1 input fields:
        nPerBlockG0, nPerBlockG1, mPerBlockG0, kPerBlock, kpack,
        numCTAs, numWaves, matrixInstrNonkdim, splitKFactor,
        numStages, wavesPerEU, gridGroupSize

    rocMLIR GemmGemm v3 output fields and how each is derived:
        mPerBlockG0      = triton.mPerBlockG0           (direct)
        mPerBlockG1      = triton.nPerBlockG1           (reordered)
        nPerBlockG0      = triton.nPerBlockG0           (reordered)
        kpackPerBlock    = triton.kPerBlock             (since kpack=1)
        mPerWave         = mPerBlockG0 / mWaves         (warpsPerTile algo)
        nPerWave         = nPerBlockG1 / nWaves         (warpsPerTile algo)
        mnPerXdl         = triton.matrixInstrNonkdim    (direct)
        kpack            = 1                            (always 1)
        splitKFactor     = triton.splitKFactor          (direct)
        scheduleVersion  = 1 if numStages<=1 else 2     (Default/DoubleBuffer)
        outputSwizzle    = 2                            (default to let rocMLIR choose)
        wavesPerEU       = triton.wavesPerEU            (direct)
        forceUnroll      = 1                            (default, Triton has no equivalent tuning param)
    """
    if len(params) != 12:
        return None

    n_per_block_g0 = params[0]
    n_per_block_g1 = params[1]
    m_per_block_g0 = params[2]
    k_per_block = params[3]
    # kpack = params[4]
    # num_ctas = params[5]
    num_waves = params[6]
    matrix_instr_nonkdim = params[7]
    split_k_factor = params[8]
    num_stages = params[9]
    waves_per_eu = params[10]
    # grid_group_size = params[11]

    kpack_per_block = k_per_block  # since rocmlirTriton kpack = 1
    m_per_wave, n_per_wave = compute_wave_distribution(
        num_waves, m_per_block_g0, n_per_block_g1, matrix_instr_nonkdim)
    schedule_version = 1 if num_stages <= 1 else 2

    # GemmGemm v3 format (see RockAttrDefs.td):
    # attn:v3:mPerBlockG0,mPerBlockG1,nPerBlockG0,kpackPerBlock,
    #         mPerWave,nPerWave,mnPerXdl,kpack,splitKFactor,
    #         scheduleVersion,outputSwizzle,wavesPerEU,forceUnroll
    return (f"attn:v3:{m_per_block_g0},"
            f"{n_per_block_g1},{n_per_block_g0},"
            f"{kpack_per_block},{m_per_wave},{n_per_wave},"
            f"{matrix_instr_nonkdim},1,{split_k_factor},"
            f"{schedule_version},2,{waves_per_eu},"
            f"1")


def convert_triton_to_rocmlir_perfconfig(perf_config):
    """Convert a rocmlirTriton perfConfig string to the equivalent rocMLIR
    perfConfig string.
    """
    try:
        parts = perf_config.split(':')
        if len(parts) != 3:
            return None

        operation = parts[0]
        params = [int(p) for p in parts[2].split(',')]

        if operation == 'gemm':
            return _convert_gemm_to_rocmlir(params)
        elif operation == 'attn':
            return _convert_attn_to_rocmlir(params)
        else:
            return None

    except (ValueError, IndexError, ZeroDivisionError) as e:
        print(f"Error converting perfConfig '{perf_config}' to rocMLIR: {e}")
        return None


def parse_results(debug_output):
    """
    This function parses the generated output file to gather the desired
    information. debug_output will contain all of the output from running
    rocmlir-driver (debug output and assembly output). It will be structured
    something like the following:
    """
    tuning_data = TuningData()

    # Look for blocksize
    blocksize_match = re.search(r'blockSize:\s*(\d+)', debug_output)
    if not blocksize_match:
        raise ValueError("Could not find blockSize in output")
    tuning_data.blocksize = int(blocksize_match.group(1))

    # Look for gridsize
    gridsize_match = re.search(r'gridSize:\s*(\d+)', debug_output)
    if not gridsize_match:
        raise ValueError("Could not find gridSize in output")
    tuning_data.gridsize = int(gridsize_match.group(1))

    # Look for waveSize
    wavesize_match = re.search(r'waveSize:\s*(\d+)', debug_output)
    if not wavesize_match:
        raise ValueError("Could not find waveSize in output")
    tuning_data.wf_per_wg = int(blocksize_match.group(1)) / int(wavesize_match.group(1))

    # Look for LDS usage from the module output (stdout).
    # In Triton's compilation model, LDS is allocated dynamically and tracked
    # via the `ttg.shared` module attribute, not in the kernel descriptor's
    # `.group_segment_fixed_size` (which is always 0 for Triton kernels).
    lds_match = re.search(r'ttg\.shared\s*=\s*(\d+)', debug_output)
    if not lds_match:
        raise ValueError("Could not find ttg.shared (LDS usage) in module output")
    tuning_data.lds_allocated = int(lds_match.group(1))

    # Look for SGPR count
    sgpr_match = re.search(r'\.sgpr_count:\s+(\d+)', debug_output)
    if not sgpr_match:
        raise ValueError("Could not find sgpr_count in output")
    tuning_data.sgpr_count = int(sgpr_match.group(1))

    # Look for VGPR count
    vgpr_match = re.search(r'\.vgpr_count:\s+(\d+)', debug_output)
    if not vgpr_match:
        raise ValueError("Could not find vgpr_count in output")
    tuning_data.vgpr_count = int(vgpr_match.group(1))

    # Look for SGPR spill count
    sgpr_spill_match = re.search(r'\.sgpr_spill_count:\s+(\d+)',
                                 debug_output)
    if not sgpr_spill_match:
        raise ValueError("Could not find sgpr_spill_count in output")
    tuning_data.sgpr_spills = int(sgpr_spill_match.group(1))

    # Look for VGPR spill count
    vgpr_spill_match = re.search(r'\.vgpr_spill_count:\s+(\d+)',
                                 debug_output)
    if not vgpr_spill_match:
        raise ValueError("Could not find vgpr_spill_count in output")
    tuning_data.vgpr_spills = int(vgpr_spill_match.group(1))

    mfma_wmma_instructions = parse_mfma_wmma_instructions(debug_output)
    if mfma_wmma_instructions:
        tuning_data.mfma_wmma_instruction = mfma_wmma_instructions[0]

    return tuning_data


def parse_perf_config(perf_config, num_cu, arch):
    """
    Parse the perfConfig string to extract tuning parameters.

    rocmlirTriton perfConfig formats (from RockDialect.cpp):

    gemm:v1:mPerBlock,nPerBlock,kPerBlock,kpack,numCTAs,numWaves,
            matrixInstrNonkdim,splitKFactor,numStages,wavesPerEU,
            gridGroupSize                                         (11 params)

    attn:v1:nPerBlockG0,nPerBlockG1,mPerBlockG0,kPerBlock,kpack,numCTAs,
            numWaves,matrixInstrNonkdim,splitKFactor,numStages,wavesPerEU,
            gridGroupSize                                         (12 params)

    Returns:
        dict: Dictionary containing parsed parameters, or None on failure.

    TODO: The format of the perfConfig string is subject to changes in the
          future, so we should at a minimum be keeping this in sync with the
          c++ code, but we should also consider making bindings to the c++ code
          that can be called from here.
    """
    try:
        # Split by ':' to separate operation, version, and parameters
        parts = perf_config.split(':')
        if len(parts) != 3:
            raise ValueError(f"Expected format 'op:vN:params', got: "
                             f"{perf_config}")

        operation = parts[0]
        version_str = parts[1]
        params_str = parts[2]

        if not version_str.startswith('v'):
            raise ValueError(f"Expected version like 'v1', got: "
                             f"{version_str}")
        version = int(version_str[1:])

        params = [int(p) for p in params_str.split(',')]
        parsed_params = {}

        # Calculate minNumWaves based on numCUs and numEUPerCU
        arch_info = amd_arch_db.lookup_arch_info(arch)
        num_eu_per_cu = getattr(arch_info, 'num_eu_per_cu')
        parsed_params['minNumWaves'] = int(num_cu) * num_eu_per_cu

        if operation == 'gemm':
            # gemm:v1:mPerBlock,nPerBlock,kPerBlock,kpack,numCTAs,numWaves,
            #         matrixInstrNonkdim,splitKFactor,numStages,wavesPerEU,
            #         gridGroupSize
            if version == 1:
                if len(params) != 11:
                    raise ValueError(f"gemm:v1 expects 11 params, got "
                                     f"{len(params)}")
                parsed_params['MPerBlock'] = params[0]
                parsed_params['NPerBlock'] = params[1]
                parsed_params['KPerBlock'] = params[2]
                parsed_params['kPack'] = params[3]
                parsed_params['numCTAs'] = params[4]
                num_waves = params[5]
                parsed_params['matrixInstrNonkdim'] = params[6]
                parsed_params['splitKFactor'] = params[7]
                parsed_params['numStages'] = params[8]
                parsed_params['wavesPerEU'] = params[9]
                parsed_params['gridGroupSize'] = params[10]

                # MNPerWave: total block tile divided by number of waves
                parsed_params['MNPerWave'] = (params[0] * params[1]) // num_waves
            else:
                raise ValueError(f"Unsupported gemm version: v{version}")

        elif operation == 'attn':
            # attn:v1:nPerBlockG0,nPerBlockG1,mPerBlockG0,kPerBlock,kpack,
            #         numCTAs,numWaves,matrixInstrNonkdim,splitKFactor,
            #         numStages,wavesPerEU,gridGroupSize
            if version == 1:
                if len(params) != 12:
                    raise ValueError(f"attn:v1 expects 12 params, got "
                                     f"{len(params)}")
                # n_per_block_g0 = params[0]
                n_per_block_g1 = params[1]
                m_per_block_g0 = params[2]
                parsed_params['KPerBlock'] = params[3]
                parsed_params['kPack'] = params[4]
                parsed_params['numCTAs'] = params[5]
                num_waves = params[6]
                parsed_params['matrixInstrNonkdim'] = params[7]
                parsed_params['splitKFactor'] = params[8]
                parsed_params['numStages'] = params[9]
                parsed_params['wavesPerEU'] = params[10]
                parsed_params['gridGroupSize'] = params[11]

                # For attention: MPerBlock = mPerBlockG0,
                # NPerBlock = nPerBlockG1 (the full N tile)
                parsed_params['MPerBlock'] = m_per_block_g0
                parsed_params['NPerBlock'] = n_per_block_g1

                # MNPerWave: block tile divided by number of waves
                parsed_params['MNPerWave'] = \
                    (m_per_block_g0 * n_per_block_g1) // num_waves
            else:
                raise ValueError(f"Unsupported attn version: v{version}")

        else:
            raise ValueError(f"Unknown operation in perfConfig: "
                             f"'{operation}'")

        return parsed_params

    except (ValueError, IndexError) as e:
        print(f"Error parsing perfConfig '{perf_config}': {e}")
        return None


def extract_mng_from_config(conf_class, operation):
    """
    Extract M, N, and G values from the testVector based on the operation type.

    Args:
        conf_class: Configuration class instance of specified operation
        operation: Operation type (e.g., 'attention', 'gemm', 'conv2d')

    Returns:
        tuple: (M, N, G) values based on operation type
    """
    try:
        if operation == Operation.ATTENTION:
            # For attention ops: M = seq_len_q, N = seq_len_k, G = g * num_heads_q
            m = conf_class.seq_len_q
            n = conf_class.seq_len_k
            g = conf_class.g * conf_class.num_heads_q

        elif operation == Operation.GEMM:
            # For GEMM ops: M = m, N = n, G = g
            m = conf_class.m
            n = conf_class.n
            g = conf_class.g

        elif operation == Operation.GEMM_GEMM:
            # For gemm+gemm ops: M = m, N = o (final output dimension), G = g
            m = conf_class.m
            n = conf_class.o
            g = conf_class.g

        elif operation == Operation.CONV:
            # For conv ops: M = k, N = batch_size * output_height * output_width,
            # G = g
            assert conf_class.direction == 'fwd', \
                "Only forward convolution (-F=1) is supported"

            # For group convolution (g > 1), validate the configuration
            if conf_class.group > 1:
                # Check that output channels and input channels are divisible by
                # group size
                assert conf_class.k % conf_class.group == 0, (
                    f"Invalid group conv config - output channels ({conf_class.k}) "
                    f"not divisible by group size ({conf_class.group})"
                )
                assert conf_class.c % conf_class.group == 0, (
                    f"Invalid group conv config - input channels ({conf_class.c}) "
                    f"not divisible by group size ({conf_class.group})"
                )

            # For group convolution: adjust k by dividing by group size. This
            # will also work in non-group conv case of g = 1
            k_per_group = conf_class.k // conf_class.group
            m = k_per_group
            g = conf_class.group
            n = conf_class.n * conf_class.ho * conf_class.wo

        else:
            print(f"Warning: Unknown operation type '{operation}'")
            return None, None, None

    except (ValueError, TypeError) as e:
        print(f"Warning: Error parsing M, N, G values from testVector: {e}")
        return None, None, None

    return m, n, g


def gather_occupancy_parameters(config, perf_config, conf_class, operation):
    '''
    This function gathers all of the parameters that are needed to calculate
    the theoretical occupancy
    '''
    num_cu = config[1]
    parsed_params = parse_perf_config(perf_config, num_cu, config[0])

    if parsed_params is None:
        return [None] * 8  # Return None values if parsing fails

    # Extract the required parameters for occupancy calculation
    [m, n, g] = extract_mng_from_config(conf_class, operation)

    m_per_block = int(parsed_params['MPerBlock'])
    n_per_block = int(parsed_params['NPerBlock'])
    mn_per_wave = int(parsed_params['MNPerWave'])
    min_num_waves = int(parsed_params['minNumWaves'])
    split_k_factor = int(parsed_params['splitKFactor'])

    return [m, n, g, m_per_block, n_per_block, mn_per_wave, min_num_waves, split_k_factor]


def compile_and_collect_data(config, operation, binaries):
    """
    Compile and collect the resulting data points that we are interested in
    """
    # Get current timestamp in a filesystem-friendly format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use the operation class
    op_type = Operation.from_name(operation)

    # Create a performance configuration class instance
    arch = config[0].split(':')[0]
    num_cu = config[1]
    test_vector = config[2]
    perf_config = config[3]
    tflops = config[4]
    # Note: num_chiplets defaults to 1 for debug db configs
    num_chiplets = 1
    conf_class = get_perf_config(op_type, test_vector, arch, num_cu, num_chiplets)
    conf_class.set_perfconfig(perf_config)

    # Compile the config. module_output (stdout) has the compiled MLIR module
    # with ttg.shared (LDS). debug_output (stderr) has debug prints and AMDGCN
    # assembly.
    module_output, debug_output = compile_config(conf_class, binaries,
                                                 timestamp)
    if isinstance(debug_output, bytes):
        debug_output = debug_output.decode('utf-8')
    if isinstance(module_output, bytes):
        module_output = module_output.decode('utf-8')

    # If the debug output is empty, then this means that the compilation
    # pipeline failed. We expect this to happen for some of the invalid configs
    if not debug_output:
        print(f"Warning: Compilation failed for config {config}. "
              "Skipping calculations.")
        return None

    # Combine both outputs so parse_results can find everything it needs
    combined_output = debug_output + "\n" + module_output
    results = parse_results(combined_output)

    # Convert rocmlirTriton perfConfig to rocMLIR perfConfig
    results.rocmlir_perfconfig = convert_triton_to_rocmlir_perfconfig(
        perf_config)

    # Convert the TFLOPs value to seconds
    results.ns = conf_class.compute_ns_from_tflops(tflops)

    # Calculate occupancy using the method in testing_metrics.py
    [m, n, g, m_per_block, n_per_block,
     mn_per_wave, min_num_waves, split_k_factor] = \
        gather_occupancy_parameters(config, perf_config, conf_class, op_type)
    # If any of the parameters are None, we cannot calculate occupancy
    if None in [m, n, g, m_per_block, n_per_block,
                mn_per_wave, min_num_waves, split_k_factor]:
        print("Warning: Could not gather all parameters for occupancy "
              "calculation for config. Skipping occupancy calculation.")
        results.occupancy = None
    elif op_type == Operation.ATTENTION:
        results.occupancy = calculate_attention_occupancy(n, g, m_per_block,
                                                          n_per_block,
                                                          mn_per_wave, min_num_waves)
    else:
        results.occupancy = calculate_gemm_occupancy(m, n, g, m_per_block, n_per_block,
                                                     mn_per_wave, min_num_waves,
                                                     split_k_factor)

    return results


def create_tsv_writer(configs, output_file):
    """
    Create and initialize a TSV writer with headers.

    Args:
        configs: Dictionary of original configuration dictionaries
        output_file: Path to the output file

    Returns:
        tuple: (file_handle, csv.DictWriter) - caller must close file_handle
    """
    # Get the original fieldnames from the first config entry
    first_config_data = next(iter(configs.values()))
    original_fieldnames = list(first_config_data.keys())
    tsv_fieldnames = original_fieldnames + NEW_DATA_FIELDNAMES

    try:
        tsvfile = open(output_file, 'w', newline='', encoding='utf-8')
        writer = csv.DictWriter(tsvfile, fieldnames=tsv_fieldnames, delimiter='\t')

        # Write the header
        writer.writeheader()
        tsvfile.flush()  # Ensure header is written immediately

        return tsvfile, writer

    except Exception as e:
        print(f"\nError creating output file: {e}")
        sys.exit(1)


def write_result_to_tsv(writer, config, config_data, result, tsvfile):
    """
    Write a single result row to the TSV file.

    Args:
        writer: csv.DictWriter instance
        config: Configuration tuple
        config_data: Original configuration dictionary
        result: TuningData result object (or None)
        tsvfile: File handle for flushing
    """
    try:
        # Start with the original config data
        row = dict(config_data)  # Copy all original fields

        # Add new tuning data
        if result is None:
            row.update({field: None for field in NEW_DATA_FIELDNAMES})
        else:
            result_dict = result.to_dict()
            row.update({field: result_dict.get(field, '') for field in NEW_DATA_FIELDNAMES})

        writer.writerow(row)
        tsvfile.flush()  # Ensure row is written immediately

    except Exception as e:
        print(f"\nError writing result to tsv: {e}")
        sys.exit(1)


def print_progress(current, total):
    """Print a progress bar to stdout."""
    prefix = "Processing Configs"
    percent = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix}: |{bar}| {current}/{total} ({percent:.1f}%)', end='',
          flush=True)
    if current == total:
        print()  # New line when complete


def main():
    """Main function to process configurations and collect tuning data."""
    parser = argparse.ArgumentParser(
        description="Compile configurations and collect tuning data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--op', "--operation", required=True,
                        help='Operation to perform (e.g., "compile")',
                        choices=['conv', 'gemm', 'attention', 'gemm_gemm'])
    parser.add_argument('config_tsv', help='Path to the tuning database file')

    args = parser.parse_args()

    # Get the paths to the rocmlir binaries
    build_bin_dir = os.path.dirname(os.path.abspath(__file__))
    rocmlir_root = os.path.dirname(build_bin_dir)
    paths = perfRunner.create_paths(None, rocmlir_root)

    # Check if the input config tsv file exists
    if not os.path.exists(args.config_tsv):
        print("Error: The specified config tsv file cannot be found.")
        return 1

    # Parse the configuration file
    configs = perfRunner.read_debug_db(args.config_tsv)

    print(f"Found {len(configs)} configurations to process")

    # Create output file and writer (streaming mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tuning_results_{timestamp}.tsv"
    tsvfile, writer = create_tsv_writer(configs, output_file)

    try:
        # Process each configuration and write immediately (no accumulation)
        config_keys = list(configs.keys())
        total_configs = len(config_keys)

        for i, config in enumerate(config_keys):
            print_progress(i, total_configs)
            config_data = configs[config]

            # Compile and collect data for this config
            metrics = compile_and_collect_data(config, args.op, paths)

            # Write result immediately
            write_result_to_tsv(writer, config, config_data, metrics, tsvfile)

        print_progress(total_configs, total_configs)
        print(f"\nResults written to {output_file}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        return 1
    finally:
        # Always close the file
        tsvfile.close()

    # If we have reached this point without crashing, it means that we have had
    # a successful run and we can return 0.
    return 0


if __name__ == "__main__":
    sys.exit(main())
