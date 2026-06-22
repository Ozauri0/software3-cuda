# Makefile for mochila_ga_cuda
# Requires: CUDA Toolkit 13.3, Visual Studio Build Tools 2022

# Compiler
NVCC = nvcc

# CUDA paths
CUDA_PATH = /c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3
CUDA_BIN = $(CUDA_PATH)/bin
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib/x64

# Visual Studio Build Tools
MSVC_PATH = /c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207
MSVC_BIN = $(MSVC_PATH)/bin/Hostx64/x64
MSVC_INC = $(MSVC_PATH)/include
MSVC_LIB = $(MSVC_PATH)/lib/x64

# Windows SDK
SDK_VER = 10.0.26100.0
SDK_BASE = /c/Program Files (x86)/Windows Kits/10
SDK_INC_UM = $(SDK_BASE)/Include/$(SDK_VER)/um
SDK_INC_UCRT = $(SDK_BASE)/Include/$(SDK_VER)/ucrt
SDK_INC_SHARED = $(SDK_BASE)/Include/$(SDK_VER)/shared
SDK_LIB_UM = $(SDK_BASE)/Lib/$(SDK_VER)/um/x64
SDK_LIB_UCRT = $(SDK_BASE)/Lib/$(SDK_VER)/ucrt/x64

# Flags
NVCC_FLAGS = -O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90
INCLUDES = -I$(CUDA_INC) -I$(MSVC_INC) -I$(SDK_INC_UM) -I$(SDK_INC_UCRT) -I$(SDK_INC_SHARED) -Isrc
LIBS = -L$(CUDA_LIB) -L$(MSVC_LIB) -L$(SDK_LIB_UM) -L$(SDK_LIB_UCRT)

# Source files
SRC_DIR = src
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
ALL_SRCS = $(CU_SRCS) $(CPP_SRCS)

# Output
TARGET = mochila_ga_cuda

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(ALL_SRCS)
	@echo "=== Compiling $(TARGET) ==="
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) -o $@ $^
	@echo "=== Build successful ==="

clean:
	rm -f $(TARGET)
	rm -rf build/

run: $(TARGET)
	./$(TARGET) --help

# Quick test build with a single file
test:
	@echo "=== Test compilation ==="
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) -o /tmp/test_cu.exe $(SRC_DIR)/main.cu
	@echo "=== Test successful ==="
