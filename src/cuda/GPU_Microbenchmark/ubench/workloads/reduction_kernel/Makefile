BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

# The reduction kernels and host driver form one executable.  The CUDA source
# includes reduction.cpp when compiled as the top-level translation unit, so do
# not compile reduction.cpp separately (that would define main() twice).
TARGET := reduction_kernel

# This machine's silicon is SM_120.  ARCH remains overridable for another GPU,
# for example: make ARCH=sm_90
ARCH ?= sm_120
CC   := nvcc

# The host driver uses helper_cuda.h and helper_functions.h from the bundled
# CUDA samples.  Use explicit code generation instead of setup_environment's
# CUDA_CPPFLAGS so the executable always contains both native SASS and PTX.
CUDA_CPPFLAGS =
CUDA_SAMPLES_INCLUDE ?= $(BASE_DIR)/../../../../NVIDIA_CUDA-11.0_Samples/common/inc
NVCC_FLAGS := $(foreach arch,$(ARCH),\
    -gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) \
    -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) \
    -std=c++17 --extended-lambda -I$(CUDA_SAMPLES_INCLUDE)

.DEFAULT_GOAL := all

all: $(TARGET)

# reduction_kernel.cu includes reduction.cpp, while reduction.cpp includes
# reduction.h.  Listing all three prerequisites keeps incremental builds sound.
$(TARGET).a: $(TARGET).cu reduction.cpp reduction.h
	$(CC) $(NVCC_FLAGS) $(CUDA_CPPFLAGS) --lib $< -o $@

$(TARGET): $(TARGET).a
	$(CC) $(NVCC_FLAGS) $< -o $@ -lcudart --cudart shared
	mkdir -p $(BIN_DIR)
	mv $@ $(BIN_DIR)

clean:
	rm -f *.o *.a *.ptx *.ptxas *.sass $(TARGET)
	rm -f $(BIN_DIR)/$(TARGET)

.PHONY: all clean
