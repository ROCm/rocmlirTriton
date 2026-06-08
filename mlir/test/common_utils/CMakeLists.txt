file(GLOB SRC_PY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

# Test-only pybind11 binding over AmdArchDb. Never linked into librockCompiler
# so downstream consumers (e.g. MIGraphX) do not pick up a pybind11 dep. The
# binding is the single source of truth for per-arch feature gating in lit,
# so it's a hard requirement whenever tests are being configured. Downstream
# builds that don't add this subdirectory remain pybind11-free.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate pybind11's CMake config through the Python interpreter rather than
# relying on the install prefix. The pybind11 PyPI package always ships its
# CMake config files inside the Python package directory, regardless of
# whether pip installed it system-wide, into a user-site (`~/.local`), or a
# venv -- so `pybind11.get_cmake_dir()` works in all CI environments without
# needing the `pybind11[global]` extra. This is the integration pattern
# documented in https://pybind11.readthedocs.io/en/stable/installing.html.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _pybind11_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_lookup_rc)
if(NOT _pybind11_lookup_rc EQUAL 0)
  message(FATAL_ERROR
    "Could not import pybind11 from ${Python3_EXECUTABLE}. "
    "Install it (e.g. `pip3 install pybind11`) and reconfigure.")
endif()
find_package(pybind11 CONFIG REQUIRED HINTS "${_pybind11_cmake_dir}")
add_subdirectory(amd_arch_db)

add_custom_target(rocmlir-common-python-test-utils
COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${SRC_PY_FILES} ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Copying common python test utils"
)
add_dependencies(rocmlir-common-python-test-utils amd_arch_db)

set(ROCMLIR_COMMON_PYTHON_TESTS_UTILS ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
