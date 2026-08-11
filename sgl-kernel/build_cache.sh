#!/bin/bash
# Cached incremental build for sgl-kernel's sm100 target.
#
# Key difference vs a clean build: this does NOT wipe build/ every time, so
# unchanged objects are reused and ccache covers the rest. First run after a
# wide change is still slow; later runs are fast.
#
# Usage:
#   ./build_cache.sh                 # incremental build of sm100/common_ops.*
#   FRESH=1 ./build_cache.sh         # wipe build/ and reconfigure from scratch
#   TARGET=all ./build_cache.sh      # build every target
#   JOBS=28 NVCC_THREADS=16 ./build_cache.sh
set -eo pipefail
cd "$(dirname "$0")"

TARGET="${TARGET:-common_ops_sm100_build}"
NVCC_THREADS="${NVCC_THREADS:-8}"
# sm100 cutlass/mxfp8 TUs are memory-hungry (~6-10G peak each), so the real cap
# is RAM, not cores. Pick the smaller of a cpu-based and a memory-based limit.
#   cpu  : ~2/3 of cores, capped at 32
#   mem  : available_GB / MEM_PER_JOB_GB
MEM_PER_JOB_GB="${MEM_PER_JOB_GB:-12}"
JOBS="${JOBS:-$(awk -v cores="$(nproc)" -v memjob="${MEM_PER_JOB_GB}" '
  BEGIN {
    cpu = int(cores*2/3); if (cpu>32) cpu=32;
  }
  /^MemAvailable:/ { avail_gb = $2/1024/1024 }
  END {
    mem = int(avail_gb/memjob);
    j = (cpu<mem?cpu:mem); if (j<1) j=1;
    print j;
  }' /proc/meminfo)}"
CUDA_NVCC="${CMAKE_CUDA_COMPILER:-/usr/local/cuda/bin/nvcc}"

if [ -z "${CMAKE_PREFIX_PATH:-}" ]; then
  CMAKE_PREFIX_PATH="$(python - <<'PY'
try:
    import torch
    print(torch.utils.cmake_prefix_path)
except Exception:
    pass
PY
)"
  export CMAKE_PREFIX_PATH
fi

# ---- ccache -----------------------------------------------------------------
export CCACHE_DIR="${CCACHE_DIR:-${HOME}/.cache/sgl-kernel/ccache}"
mkdir -p "${CCACHE_DIR}"
if ! command -v ccache >/dev/null 2>&1; then
  echo "ccache not found, installing via conda-forge..."
  conda install -y -c conda-forge ccache
fi
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_COMPILERCHECK=content
export CCACHE_COMPRESS=true
export CCACHE_SLOPPINESS=file_macro,time_macros,include_file_mtime,include_file_ctime
ccache -M 50G >/dev/null
ccache -z >/dev/null

echo "TARGET=${TARGET}  JOBS=${JOBS} (nproc=$(nproc), mem/job=${MEM_PER_JOB_GB}G)  NVCC_THREADS=${NVCC_THREADS}  CCACHE_DIR=${CCACHE_DIR}"

# ---- configure (only when needed) -------------------------------------------
if [ "${FRESH:-0}" = "1" ]; then
  rm -rf build
fi
# Reconfigure if no build tree, or if the existing one lacks ccache launchers.
if [ ! -f build/build.ninja ] || ! grep -q "ccache" build/build.ninja 2>/dev/null; then
  cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${CUDA_NVCC}" \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}" \
    -DSGL_KERNEL_COMPILE_THREADS="${NVCC_THREADS}"
fi

# ---- build ------------------------------------------------------------------
ninja -C build -j "${JOBS}" -l "${JOBS}" "${TARGET}"
ccache -s
