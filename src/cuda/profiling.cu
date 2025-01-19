#include <iostream>
#include <cuda_runtime.h>

void printDeviceProperties(const cudaDeviceProp& prop, int device) {
    std::cout << "GPU #" << device << ": " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max thread dimensions (x, y, z): "
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << std::endl;
    std::cout << "  Max grid dimensions (x, y, z): "
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << std::endl;
    std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "  Total constant memory: " << prop.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Maximum texture size (1D): " << prop.maxTexture1D << std::endl;
    std::cout << "  Maximum texture size (2D): " << prop.maxTexture2D[0] << " x " << prop.maxTexture2D[1] << std::endl;
    std::cout << "  Maximum texture size (3D): " << prop.maxTexture3D[0] << " x " << prop.maxTexture3D[1] << " x " << prop.maxTexture3D[2] << std::endl;
    std::cout << "  Concurrent kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "  ECC enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
    std::cout << "  Unified addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
    std::cout << "  PCI Bus ID: " << prop.pciBusID << std::endl;
    std::cout << "  PCI Device ID: " << prop.pciDeviceID << std::endl;
    std::cout << "  PCI Domain ID: " << prop.pciDomainID << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
}

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get the number of devices: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Number of CUDA-capable devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);

        if (err != cudaSuccess) {
            std::cerr << "Failed to get properties for device " << i << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        printDeviceProperties(prop, i);
    }

    return 0;
}

/*
Number of CUDA-capable devices: 1
GPU #0: NVIDIA A30
  Compute capability: 8.0
  Total global memory: 24169 MB
  Shared memory per block: 48 KB
  Registers per block: 65536
  Warp size: 32
  Max threads per block: 1024
  Max thread dimensions (x, y, z): 1024, 1024, 64
  Max grid dimensions (x, y, z): 2147483647, 65535, 65535
  Clock rate: 1440 MHz
  Memory clock rate: 1215 MHz
  Memory bus width: 3072 bits
  Total constant memory: 64 KB
  Multiprocessors: 56 => in 8.0 architectures there are 64 CUDA cores per streaming multiprocessor => 3584 cores in total
  Maximum texture size (1D): 131072
  Maximum texture size (2D): 131072 x 65536
  Maximum texture size (3D): 16384 x 16384 x 16384
  Concurrent kernels: Yes
  ECC enabled: Yes
  Unified addressing: Yes
  PCI Bus ID: 23
  PCI Device ID: 0
  PCI Domain ID: 0
-----------------------------------------
Most interesting specs to me:
NVIDIA A30
    Compute capability: 8.0
    Total global memory: 24169 MB
    Total constant memory: 64 KB
    Shared memory per block: 48 KB
    Registers per block: 65536
    Warp size: 32
    Multiprocessors: 56
    CUDA cores per SM: 64
    Total CUDA cores: 3584
    Concurrent kernels: Yes
-----------------------------------------
*/