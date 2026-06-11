#!/usr/bin/env bash
# Build rocmlirTriton via cget. LLVM/MLIR is built once in a persistent tree
# ($LLVM_SRC/build) and installed into $PREFIX/llvm; each run incrementally
# rebuilds + reinstalls only what changed there. rocmlirTriton itself is then
# (re)configured and built incrementally in cget's persistent per-source build
# dir, relinking against the refreshed LLVM. Editing LLVM *or* rocmlirTriton
# source and re-running this script picks up both incrementally.
set -uo pipefail

SRC=/mnt/data/pamartin/rocmlirTriton
# cget prefix: all build state (cget/, installed bin/lib/include, and the
# per-source build trees) lives here. Kept under $SRC/build, which .gitignore
# already excludes (/build*).
PREFIX="$SRC/build"

# Ninja gives near-instant no-op/incremental builds; the Unix Makefiles
# generator re-scans every target's dependencies on each invocation, which costs
# ~30s even for a one-file change on a tree this size.
GENERATOR="Ninja"

# external/triton must be a symlink to the patched rocmlirTriton-triton tree
# (cmake/triton.cmake does add_subdirectory on it). git submodule/checkout
# operations periodically reset this path to an empty submodule placeholder dir,
# which makes configure fail with "does not contain a CMakeLists.txt file".
# Recreate the symlink whenever it has been clobbered.
TRITON_SRC=/mnt/data/pamartin/rocmlirTriton-triton
if [ ! -f "$SRC/external/triton/CMakeLists.txt" ]; then
  echo "Restoring external/triton symlink -> $TRITON_SRC"
  rm -rf "$SRC/external/triton"
  ln -s "$TRITON_SRC" "$SRC/external/triton"
fi

cd "$SRC"

# LLVM/MLIR: persistent, incrementally-rebuildable build.
# rocmlirTriton links LLVM/MLIR as prebuilt static libs. Building them via
# `cget install` compiles in a throwaway temp dir that cget deletes afterwards,
# so an edit to the LLVM source can never be picked up incrementally (and there
# is no build dir left to rebuild). Instead we keep our own persistent LLVM
# build tree and install from it on every run: a near-instant no-op when nothing
# changed, an incremental recompile + reinstall when LLVM source is edited. The
# rocMLIR build below then relinks against the refreshed artifacts.
LLVM_SRC=/mnt/data/pamartin/rocmlirTriton-llvm
LLVM_BUILD="$LLVM_SRC/build"
LLVM_INSTALL="$PREFIX/llvm"

# Configure once. Flags mirror $TRITON_SRC/requirements.txt (the old cget
# TritonLLVM recipe) so the artifacts stay ABI-compatible with rocmlirTriton.
if [ ! -f "$LLVM_BUILD/build.ninja" ]; then
  echo "Configuring persistent LLVM build at $LLVM_BUILD (one-time)"
  cmake -S "$LLVM_SRC" -B "$LLVM_BUILD" -G "$GENERATOR" \
    -D CMAKE_INSTALL_PREFIX="$LLVM_INSTALL" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=clang-20 \
    -D CMAKE_CXX_COMPILER=clang++-20 \
    -D CMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
    -D CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
    -D CMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
    -D LLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" \
    -D LLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
    -D LLVM_ENABLE_ASSERTIONS=ON \
    -D MLIR_ENABLE_ROCM_RUNNER=ON \
    -D LLVM_OPTIMIZED_TABLEGEN=ON \
    -D MLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -D LLVM_ENABLE_ZSTD=OFF \
    -D LLVM_ENABLE_LLD=ON \
    -D LLVM_INSTALL_UTILS=ON
fi

# Incrementally (re)build and install LLVM/MLIR. First run is a long full build;
# afterwards this is ~instant when nothing changed, or a small recompile +
# reinstall when LLVM source was edited.
echo "Building + installing LLVM/MLIR (incremental)"
ninja -C "$LLVM_BUILD" install

# cget creates the per-source builder dir *before* it installs dependencies and
# configures. If a run is interrupted in between (e.g. during a long dependency
# build), it leaves an empty, un-configured builder dir behind. On the next run
# cget sees that dir exists, assumes it is already configured, skips configure,
# and then fails with "<dir>/build is not a directory". Clear such stale
# half-states, and also any dir configured with a different generator (so a
# Make->Ninja switch reconfigures cleanly). A dir already configured with the
# desired generator is left intact so incremental rebuilds still work.
BUILD_ROOT="$PREFIX/cget/build"
case "$GENERATOR" in
  Ninja*) GEN_FILE="build.ninja" ;;
  *)      GEN_FILE="Makefile" ;;
esac
for d in "$BUILD_ROOT"/_url_*; do
  [ -d "$d" ] || continue
  cache="$d/build/CMakeCache.txt"
  # A complete, usable configure leaves BOTH CMakeCache.txt and the generator
  # build file. A failed configure (e.g. a FATAL_ERROR) writes the cache early
  # but never the build file -- cget would then skip configure and run the
  # generator against a dir with no build.ninja/Makefile. Treat that as stale.
  if [ ! -f "$cache" ] || [ ! -f "$d/build/$GEN_FILE" ]; then
    echo "Removing stale/incompletely-configured cget builder dir: $d"
    rm -rf "$d"
  elif ! grep -q "^CMAKE_GENERATOR:INTERNAL=${GENERATOR}$" "$cache"; then
    echo "Removing cget builder dir configured with a different generator: $d"
    rm -rf "$d"
  fi
done

# NOTE: cget only runs cmake *configure* when the build dir does not yet exist.
# Once configured, re-running this script is incremental: 'cmake --build' still
# re-runs configure automatically when a CMakeLists.txt changes, so source and
# CMake edits are picked up without a full rebuild.
#
# Only if configure itself is broken (e.g. a half-configured dir with a cache
# but no Makefile), wipe it once with:  cget -p "$PREFIX" build -C -y

# Build with clang + lld (the tree uses clang-only pragmas/flags; system GCC
# trips -Werror=unknown-pragmas/shadow/missing-declarations).  Matches the
# canonical AMDMIGraphX/rocmlirTriton/cmake.sh.
#
# ROCMLIRTRITON_SKIP_SUBMODULE_UPDATE=ON: external/triton is a symlink to the
# pre-patched rocmlirTriton-triton tree, so the configure-time
# 'git submodule update' must not run.
#
# LLVM_SYSPATH=$LLVM_INSTALL: Triton's CMake only honors an explicit
# LLVM_SYSPATH -- it never derives one from CMAKE_PREFIX_PATH. Without this,
# Triton's build_helpers.py ignores our LLVM and downloads its pinned prebuilt
# LLVM tarball. The -D flags here propagate to the Triton dependency configure,
# so this points Triton at the LLVM we just built/installed.
# MLIR_DIR: cmake/triton.cmake locates LLVM/MLIR only via MLIR_DIR /
# LLVM_LIBRARY_DIR / a Triton-built llvm-project tree -- it does NOT consult
# CMAKE_PREFIX_PATH or LLVM_SYSPATH. Without this it can't see our MLIR and
# falls back to scripts/build-llvm.sh (which clones upstream LLVM). Point it
# straight at the persistent LLVM install.
# BUILD_SHARED_LIBS=Off: rocMLIR's library graph has inter-library symbol
# references that only resolve under static linking. With shared libs (the
# default on a fresh configure) plus the tree's -Wl,-z,defs, linking the Rock/
# Triton .so's fails with many "undefined symbol" errors. Static matches the
# known-good configuration.
time cget -v -p "$PREFIX" build \
  -G "$GENERATOR" \
  -D ROCMLIRTRITON_SKIP_SUBMODULE_UPDATE=ON \
  -D MLIR_DIR="$LLVM_INSTALL/lib/cmake/mlir" \
  -D LLVM_SYSPATH="$LLVM_INSTALL" \
  -D BUILD_SHARED_LIBS=Off \
  -D CMAKE_C_COMPILER=clang-20 \
  -D CMAKE_CXX_COMPILER=clang++-20 \
  -D CMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -D CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -D CMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld

# cget's real build dir is <prefix>/cget/build/_url_<base64-of-source>/build --
# deterministic but ugly. Point a stable symlink at it so you can run
# `ninja -C "$BUILD_DIR"` (or `cmake --build`) without copying the base64 path.
BUILD_LINK="$PREFIX/rocmlir"
REAL_BUILD_DIR="$(cget -p "$PREFIX" build -P 2>/dev/null)"
if [ -n "$REAL_BUILD_DIR" ] && [ -d "$REAL_BUILD_DIR" ]; then
  ln -sfn "$REAL_BUILD_DIR" "$BUILD_LINK"
  echo
  echo "Build dir:       $REAL_BUILD_DIR"
  echo "Stable symlink:  $BUILD_LINK"
  echo "Tip: export BUILD_DIR=\"$BUILD_LINK\"   # then: ninja -C \"\$BUILD_DIR\" rocmlir-opt"
fi
