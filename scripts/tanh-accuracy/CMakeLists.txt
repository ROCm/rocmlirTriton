# Accuracy measurement for the tanh lowering in rock-legalize-math-for-triton.
#
# These produce the numbers quoted in the emitApproxTanh comment in
# mlir/lib/Dialect/Rock/Transforms/LegalizeMathForTriton.cpp. Re-run them if
# that lowering changes.
#
# Configured on its own rather than wired into the rocMLIR build: scripts/ is
# not part of that build, and these need both a HIP compiler and a real GPU of
# the architecture being measured, neither of which rocMLIR itself requires.
#
#   cmake -S scripts/tanh-accuracy -B /tmp/tanh-accuracy \
#         -DCMAKE_HIP_ARCHITECTURES=gfx1100
#   cmake --build /tmp/tanh-accuracy
#   /tmp/tanh-accuracy/tanh-ulp-f32          # ~6s, sweeps all 2^32 patterns
#   /tmp/tanh-accuracy/tanh-ulp-f16-bf16     # instant, 65536 patterns each
#   /tmp/tanh-accuracy/tanh-ulp-f8-f4        # instant, 256 or 16 patterns each
#
# All three exit non-zero if any input saturates to something other than +/-1
# or produces a NaN, so they can be run as a check and not just read.

cmake_minimum_required(VERSION 3.21)

# Must be set before project(), which is what bakes in the offload target. The
# measurement is architecture specific: it includes the error of that GPU's
# v_exp/v_rcp units, so the target has to match the GPU it will run on.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1100)
endif()

project(rocmlir-tanh-accuracy LANGUAGES HIP)

foreach(tool tanh-ulp-f32 tanh-ulp-f16-bf16 tanh-ulp-f8-f4)
  add_executable(${tool} ${tool}.hip)
  # No fast math, deliberately. The baseline column is __ocml_tanh_f32 reached
  # through tanhf, and the approximation is spelled out one hardware
  # instruction at a time; fast math would rewrite both and the comparison
  # would stop measuring what it claims to.
  #
  # Denormal flushing is not about speed here, it is matching the kernel under
  # test: rocMLIR's Triton pipeline emits .amdhsa_float_denorm_mode_32 0, and
  # hipcc would otherwise default these to 3. It happens to leave every number
  # in the table unchanged, because the approximation already returns exactly
  # zero for arguments that small either way, but relying on that would be
  # relying on luck. f16 and f64 denormals stay enabled in both.
  #
  # The hip* return codes are ignored throughout. There is no recovery path in a
  # one-shot harness and a failure shows up immediately as a nonsense table, so
  # the alternative is a (void) cast on every call for no added safety.
  target_compile_options(${tool} PRIVATE -O2 -fno-fast-math
                                         -fgpu-flush-denormals-to-zero
                                         -Wno-unused-result)
endforeach()
