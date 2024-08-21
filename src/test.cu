#include <cuda.h>
using namespace std;

__global__ void fun() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("i = %d", i);
}

int main() {
    cudaFree(0);
    int threads_per_block = 8;
    int number_of_blocks = 4;
    fun<<< number_of_blocks, threads_per_block >>>();
    cudaDeviceSynchronize();

    return 0;
}