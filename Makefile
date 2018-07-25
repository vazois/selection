CC=g++
NVCC=/usr/local/cuda-9.0/bin/nvcc
NVCC_INCLUDE=-I../TopK/cub-1.7.4/ -I/usr/local/cuda/include/
NVCC_LIBS=-L/usr/local/cuda-9.0/lib64/

#GPU CONFIGURATION
GC_MAIN=main.cu
GC_EXE=gpu_run
#NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true
ARCH = -gencode arch=compute_61,code=sm_61
#ARCH = -gencode arch=compute_35,code=sm_35

all: gpu_cc

gpu_cc:
	$(NVCC) $(NVCC_LIBS) -std=c++11 $(ARCH) $(GC_MAIN) -o $(GC_EXE) $(NVCC_INCLUDE)
	
clean:
	rm -rf $(GC_EXE)  
