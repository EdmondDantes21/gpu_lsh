# Compiler and flags
NVCC = nvcc
GXX = g++
NVCC_FLAGS = -rdc=true -Xcompiler -fopenmp
GXX_FLAGS = -fopenmp

# Source files
CU_SRC = marzola.cu
CPP_SRC = lsh.cpp point.cpp index.cpp index_CPU.cpp

# Output binaries
CU_OUTPUT = marzola.out
CPP_OUTPUT = lsh.out

# Default target
all: $(CU_OUTPUT) $(CPP_OUTPUT)

# Rule to compile the CUDA program
$(CU_OUTPUT): $(CU_SRC)
	$(NVCC) $(NVCC_FLAGS) $(CU_SRC) -o $(CU_OUTPUT)

# Rule to compile the C++ program
$(CPP_OUTPUT): $(CPP_SRC)
	$(GXX) $(GXX_FLAGS) $(CPP_SRC) -o $(CPP_OUTPUT)

# Clean up compiled files
clean:
	rm -f $(CU_OUTPUT) $(CPP_OUTPUT)

# Phony targets
.PHONY: all clean
