#!/bin/bash
# build.sh - Script de compilación para mochila_ga_cuda
# Configura el entorno VS Build Tools + CUDA y compila

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$PROJECT_DIR/src"
BUILD_DIR="$PROJECT_DIR/build"
OUTPUT="$PROJECT_DIR/mochila_ga_cuda"

# CUDA paths
CUDA_BIN="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin"
CUDA_INC="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/include"
CUDA_LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/lib/x64"

# Visual Studio Build Tools paths
MSVC_BIN="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
MSVC_INC="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/include"
MSVC_LIB="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/lib/x64"

# Windows SDK paths
SDK_VER="10.0.26100.0"
SDK_BASE="/c/Program Files (x86)/Windows Kits/10"
SDK_LIB_UM="$SDK_BASE/Lib/$SDK_VER/um/x64"
SDK_LIB_UCRT="$SDK_BASE/Lib/$SDK_VER/ucrt/x64"
SDK_INC_UM="$SDK_BASE/Include/$SDK_VER/um"
SDK_INC_UCRT="$SDK_BASE/Include/$SDK_VER/ucrt"
SDK_INC_SHARED="$SDK_BASE/Include/$SDK_VER/shared"

# Build directory
mkdir -p "$BUILD_DIR"

echo "=== Compiling mochila_ga_cuda ==="
echo "CUDA: $CUDA_BIN"
echo "MSVC: $MSVC_BIN"
echo ""

# Compile with nvcc
# nvcc needs cl.exe in PATH and the include/lib paths set
export PATH="$CUDA_BIN:$MSVC_BIN:$PATH"

# Collect all source files
CU_FILES=$(find "$SRC_DIR" -name "*.cu" 2>/dev/null)
CPP_FILES=$(find "$SRC_DIR" -name "*.cpp" 2>/dev/null)

echo "Source files:"
echo "$CU_FILES"
echo "$CPP_FILES"
echo ""

# Build nvcc command with all include paths
nvcc -O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 \
  -I"$CUDA_INC" \
  -I"$MSVC_INC" \
  -I"$SDK_INC_UM" \
  -I"$SDK_INC_UCRT" \
  -I"$SDK_INC_SHARED" \
  -I"$SRC_DIR" \
  -L"$CUDA_LIB" \
  -L"$MSVC_LIB" \
  -L"$SDK_LIB_UM" \
  -L"$SDK_LIB_UCRT" \
  -o "$OUTPUT" \
  $CU_FILES $CPP_FILES

echo ""
echo "=== Build successful ==="
echo "Output: $OUTPUT"
echo ""
echo "Run with:"
echo "  $OUTPUT --help"
