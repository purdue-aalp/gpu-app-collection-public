BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

# Build all mbarrier microbenchmarks in this directory.
ALL_BENCHES := mbarrier mbarrier_reduction mbarrier_reduction2 mbarrier_test

ARCH ?= sm_80 sm_90a sm_100a sm_101 sm_120
CC   := nvcc

# Unset CUDA_CPPFLAGS from setup_environment; use explicit codegen here.
CUDA_CPPFLAGS=

# Generate both SASS and PTX.
NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) -std=c++17

# Cluster-scope mbarrier (mapa, cluster launch) needs sm_90+: build Hopper only.
CLUSTER_FLAGS := -gencode=arch=compute_90a,code=sm_90a -gencode=arch=compute_90a,code=compute_90a -std=c++17

.DEFAULT_GOAL := all

all: $(ALL_BENCHES) mbarrier_cluster

%.a: %.cu
	$(CC) $(NVCC_FLAGS) $(CUDA_CPPFLAGS) --lib $< -o $@

$(ALL_BENCHES): %: %.a
	$(CC) $(NVCC_FLAGS) $< -o $@ -lcudart --cudart shared
	mv $@ $(BIN_DIR)

mbarrier_cluster: mbarrier_cluster.cu
	$(CC) $(CLUSTER_FLAGS) $< -o $@ -lcudart --cudart shared
	mv $@ $(BIN_DIR)

clean:
	rm -f *.o *.a *.ptx *.ptxas *.sass $(ALL_BENCHES) mbarrier_cluster

.PHONY: all clean
