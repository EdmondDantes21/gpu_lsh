#ifndef COMMON_HPP
#define COMMON_HPP

#define DIMENSIONS 4
#define N_HYPERPLANES 128
#define BUCKET_SIZE 4

// Macro for error checking
#define CHECK_CUDA(call) do {                              \
    cudaError_t err = call;                               \
    if (err != cudaSuccess) {                             \
        std::cerr << "CUDA error in " << __FILE__        \
                  << " at line " << __LINE__             \
                  << ": " << cudaGetErrorString(err)    \
                  << std::endl;                          \
        exit(err);                                        \
    }                                                      \
} while (0)

#endif