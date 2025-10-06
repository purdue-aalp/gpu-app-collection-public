# Makefile for CUTLASS Hopper Tensor Core Examples
# Ported from CUTLASS examples/cute/tutorial/hopper

# Compiler
NVCC := nvcc

# CUDA architecture for Hopper (SM 90)
CUDA_ARCH := 90a

# Base paths
GPU_APP_ROOT := ../../..
CUTLASS_INCLUDE := $(GPU_APP_ROOT)/include
COMMON_INCLUDE := $(GPU_APP_ROOT)/common
TOOLS_UTIL_INCLUDE := $(GPU_APP_ROOT)/include/cutlass_tools_util_include

# Compiler flags
NVCC_FLAGS := -std=c++17 \
              --expt-relaxed-constexpr \
              -O3 \
              -DNDEBUG \
              -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
              -DCUTLASS_TEST_LEVEL=0 \
              -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
              --generate-code=arch=compute_$(CUDA_ARCH),code=[sm_$(CUDA_ARCH),compute_$(CUDA_ARCH)] \
              -Xcompiler=-fno-strict-aliasing \
              -forward-unknown-to-host-compiler

# Include paths
INCLUDE_DIRS := -I$(CUTLASS_INCLUDE) \
                -I$(COMMON_INCLUDE) \
                -I$(TOOLS_UTIL_INCLUDE)

# Targets
TARGETS := wgmma_sm90 wgmma_tma_sm90

.PHONY: all clean

all: $(TARGETS)

wgmma_sm90: wgmma_sm90.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $< -o $@

wgmma_tma_sm90: wgmma_tma_sm90.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $< -o $@

clean:
	rm -f $(TARGETS) *.o

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build all examples (default)"
	@echo "  wgmma_sm90       - Build WGMMA example"
	@echo "  wgmma_tma_sm90   - Build WGMMA with TMA example"
	@echo "  clean            - Remove built binaries"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA Toolkit with SM 90 (Hopper) support"
	@echo "  - GPU with compute capability 9.0 or higher"
