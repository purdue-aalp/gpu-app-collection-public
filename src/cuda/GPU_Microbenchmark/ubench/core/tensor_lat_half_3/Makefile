# Standalone Makefile for CUTLASS CuTE tcgen05.mma Tensor Latency Benchmark (SM100)
# Does not use common.mk to avoid environment variable conflicts

# Compiler
NVCC := /usr/local/cuda-12.8/bin/nvcc

# Source and executable
SRC := tensor_lat_half.cu
EXE := tensor_lat_half

# Output directory
BIN_DIR := ../../../bin

# CUTLASS include paths (7 levels up to gpu-app-collection)
GPU_APP_ROOT := ../../../../../..
CUTLASS_INCLUDE := -I$(GPU_APP_ROOT)/include
COMMON_INCLUDE := -I$(GPU_APP_ROOT)/include/cutlass_common
TOOLS_UTIL_INCLUDE := -I$(GPU_APP_ROOT)/include/tools_util_include

# Architecture flags - CUTLASS CuTE tcgen05.mma requires SM100a (Blackwell)
ARCH_FLAGS := -gencode=arch=compute_100a,code=\"sm_100a,compute_100a\"

# CUTLASS/CuTE compiler flags for SM100
CUTLASS_FLAGS := -std=c++17 \
                 --expt-relaxed-constexpr \
                 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
                 -DCUTLASS_ARCH_MMA_SM100_SUPPORTED \
				 -DCUTLASS_ARCH_MMA_SM100A_ENABLED \
                 -DCUTLASS_DISABLE_SYNCLOG \
                 -DTUNER \
                 -lineinfo

# Combine all flags
NVCC_FLAGS := $(ARCH_FLAGS) $(CUTLASS_FLAGS) $(CUTLASS_INCLUDE) $(COMMON_INCLUDE) $(TOOLS_UTIL_INCLUDE)

# Targets
.PHONY: all release clean run

all: release

release: $(EXE)
	@mkdir -p $(BIN_DIR)
	cp $(EXE) $(BIN_DIR)/

$(EXE): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(EXE) -lcudart

clean:
	rm -f *.o $(EXE)

run: $(EXE)
	./$(EXE)
