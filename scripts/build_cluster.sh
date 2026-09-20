#!/usr/bin/env bash
# Build a cluster-compatible GPUMD package on the CentOS 7 cluster.
#
# The defaults match the current cluster installation. Override them when
# building on another compatible system, for example:
#   GPUMD_CUDA_ARCHITECTURES=70 scripts/build_cluster.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
SOURCE_ROOT=${GPUMD_SOURCE_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}

CUDA_HOME=${CUDA_HOME:-/home/m9n0o/opt/cuda-12.6.0}
GCC_HOME=${GPUMD_GCC_HOME:-/home/software/GCC/gcc-11.2.0}
CMAKE_BIN=${GPUMD_CMAKE:-/home/m9n0o/opt/cmake-3.31.8/bin/cmake}
ARCHITECTURES=${GPUMD_CUDA_ARCHITECTURES:-70\;80}
BUILD_ROOT=${GPUMD_BUILD_DIR:-$SOURCE_ROOT/build/cluster-cuda}
PACKAGE_ROOT=${GPUMD_PACKAGE_DIR:-$SOURCE_ROOT/build/gpumd-cluster}
BUILD_PARALLEL=${GPUMD_BUILD_PARALLEL:-2}

for required in \
  "$CMAKE_BIN" \
  "$CUDA_HOME/bin/nvcc" \
  "$GCC_HOME/bin/gcc" \
  "$GCC_HOME/bin/g++"; do
  if [[ ! -x "$required" ]]; then
    echo "Required tool is missing or not executable: $required" >&2
    exit 1
  fi
done

export PATH="$CUDA_HOME/bin:$GCC_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$GCC_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "Source:         $SOURCE_ROOT"
echo "Build directory: $BUILD_ROOT"
echo "Package:        $PACKAGE_ROOT"
echo "CUDA:           $CUDA_HOME"
echo "Host compiler:  $GCC_HOME"
echo "Architectures:  $ARCHITECTURES"

"$CMAKE_BIN" \
  -S "$SOURCE_ROOT" \
  -B "$BUILD_ROOT" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHITECTURES" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$GCC_HOME/bin/g++" \
  -DCMAKE_C_COMPILER="$GCC_HOME/bin/gcc" \
  -DCMAKE_CXX_COMPILER="$GCC_HOME/bin/g++"

if [[ ${GPUMD_CONFIGURE_ONLY:-0} == 1 ]]; then
  echo "Cluster configuration completed at $BUILD_ROOT"
  exit 0
fi

"$CMAKE_BIN" --build "$BUILD_ROOT" --parallel "$BUILD_PARALLEL" --target gpumd nep

mkdir -p "$PACKAGE_ROOT/bin"
install -m 0755 "$BUILD_ROOT/gpumd" "$PACKAGE_ROOT/bin/gpumd"
install -m 0755 "$BUILD_ROOT/nep" "$PACKAGE_ROOT/bin/nep"

# Runtime NEP specialization needs sources matching the executable commit.
mkdir -p "$PACKAGE_ROOT/src"
cp -a "$SOURCE_ROOT/src/." "$PACKAGE_ROOT/src/"

cat > "$PACKAGE_ROOT/env.sh" <<EOF
export GPUMD_HOME="$PACKAGE_ROOT"
export CUDA_HOME="$CUDA_HOME"
export GPUMD_SRC="\$GPUMD_HOME/src"
export PATH="\$GPUMD_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$GCC_HOME/lib64:\${LD_LIBRARY_PATH:-}"
EOF

{
  echo "git_sha=$(git -C "$SOURCE_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "host_os=$(uname -srm)"
  echo "glibc=$(ldd --version | head -1)"
  echo "cuda=$("$CUDA_HOME/bin/nvcc" --version | tail -1)"
  echo "gcc=$("$GCC_HOME/bin/g++" --version | head -1)"
  echo "cmake=$("$CMAKE_BIN" --version | head -1)"
  echo "cuda_architectures=$ARCHITECTURES"
} > "$PACKAGE_ROOT/build-info.txt"

echo "Cluster package created at $PACKAGE_ROOT"
