#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "point_gpu.hpp"
#include "common.hpp"

using namespace std;

__constant__ float d_hyperplanes[DIMENSIONS * N_HYPERPLANES];   // pointer to random hyperplanes stored on device
__constant__ unsigned long long int prime;  // 64-bit prime often used in hashing

/**
 * @brief hahses a signature to an unsigned int representing whether the key is left/right of each hyperplane
 * @param key is the signature
 * @return the hashed signature
 * @note __forceinline__ strongly suggests to inline this function due to its simplicity
 */
__device__ __forceinline__ unsigned int hash_signature(unsigned long long int key) {
    key = (key ^ (key >> 30)) * prime; // Mix upper and lower bits
    key = (key ^ (key >> 27)) * prime; // Further mix bits
    key = key ^ (key >> 31);           // Final mixing
    return static_cast<unsigned int>(key); // Cast to unsigned int (lower 32 bits)
}

/**
 * @brief twice the bucket capacity
 * @param bucket is a pointer to the bucket
 * @param current_size is the current size of the bucket
 */
__device__ void resize(int **bucket, int current_size) {
    // allocate new bucket with twice the size as the old one (requires compute capability of 3.0 or above)
    int *new_bucket = (int*) malloc(current_size * 2 * sizeof(int));

    // copy data from old bucket to new bucket
    for(int i = 0; i < current_size; i++)
        new_bucket[i] = *(bucket)[i];
    
    cudaFree(*bucket);   // free old bucket memory
    *bucket = new_bucket; // bucket now points to the new bucket data
}

__device__ inline bool is_power_of_two(unsigned int n) {
    return (n & (n - 1)) == 0;
}

__device__ unsigned long long int signature_gpu(float* points, int point_number) {
    unsigned long long int sig = 0;
    unsigned int exponent = 0;
    float base = 0.0;

    for (int i = 0; i < N_HYPERPLANES; i++) {
        base = 0.0;
        #pragma unroll // when DIMENSIONS IS SMALL
        for (int j = 0; j < DIMENSIONS; j++)
            // base += points[point_number + j] * hyperplanes[DIMENSIONS * i + j];
            base = __fmaf_rn(points[point_number + j], d_hyperplanes[DIMENSIONS * i + j], base);

        if (base >= 0.0) 
            sig += (1ULL << exponent);

        exponent++;
    }   
    return sig;
}

__global__ void add_device(float *points, int **buckets, unsigned long long int *signatures, unsigned int *bucket_size, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_count = gridDim.x * blockDim.x;
    
    for (int i = tid; i < n; i += thread_count) {
        signatures[i] = signature_gpu(points, i);  // calculate point signature, i.e. whether it is left/right of each hyperplane
        unsigned int bucket_index = (hash_signature(signatures[i]) & ((n-1) / thread_count)) + n / thread_count * tid ; // bucket index indicates in which bucket to put this point. Note that we store the index of the point rather than the point itself
    
        if (bucket_size[bucket_index] >= BUCKET_SIZE && is_power_of_two(bucket_size[bucket_index])) // each time a bucket is full we double its capacity
            resize(&buckets[bucket_index], bucket_size[bucket_index]);
        buckets[bucket_index][bucket_size[bucket_index]] = i; // put the index of the i-th point in the right bucket
        bucket_size[bucket_index]++;
    }
}

/**
 * @brief search n points in the index
 * @param points is the set of points to search
 * @param buckets is the bucket indexes
 * @param result are the resulting indexes
 * @param n the number of points to search
 */
__global__ void search(float *points, int **buckets, unsigned int *bucket_size, unsigned int* result, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_count = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += thread_count) {
        unsigned long long int signature = signature_gpu(points, i);
        result[i] = hash_signature(signature);
    }
}

float *generate_random_hyperplanes(int d, int nbits) {
    float *hyperplanes = new float[d * nbits];

    srand(time(NULL));
    const float lower_bound = -1.0;
    const float upper_bound = 1.0;

    for (int i = 0; i < nbits; i++) {
        for (int j = 0; j < d; j++) {
            hyperplanes[i * d + j] = lower_bound + (upper_bound - lower_bound) * ((float) rand() / RAND_MAX);
        }
    }
    return hyperplanes;
}

float* generate_random_points() {
    float * points = new float[N * DIMENSIONS];

    srand(time(NULL));
    const float lower_bound = -1000.0;
    const float upper_bound = 1000.0;
    
    for (int i = 0; i < N * DIMENSIONS; i++) {
        points[i] = lower_bound + (upper_bound - lower_bound) * ((float) rand() / RAND_MAX);
    }
    return points;
}

void allocateMatrixOnDevice(int*** d_matrix, int rows, int cols) {
    int** h_row_ptrs = new int*[rows]; // Host array of pointers

    // Allocate device memory for the array of pointers
    int** d_row_ptrs;
    cudaMalloc((void**)&d_row_ptrs, rows * sizeof(int*));

    // Allocate memory for each row on the device
    for (int i = 0; i < rows; ++i) {
        int* d_row;
        cudaMalloc((void**)&d_row, cols * sizeof(int));
        h_row_ptrs[i] = d_row; // Store row pointer in host array
    }

    // Copy the row pointers from host to device
    cudaMemcpy(d_row_ptrs, h_row_ptrs, rows * sizeof(int*), cudaMemcpyHostToDevice);

    // Copy the device row pointers back to the device
    for (int i = 0; i < rows; ++i) {
        cudaMemcpy(&(d_row_ptrs[i]), &h_row_ptrs[i], sizeof(int*), cudaMemcpyHostToDevice);
    }

    *d_matrix = d_row_ptrs; // Assign the allocated device matrix
    delete[] h_row_ptrs; // Free host row pointers
}

int main() {
    /* POINTERS TO HOST */
    float *h_hyperplanes = generate_random_hyperplanes(DIMENSIONS, N_HYPERPLANES); // host random hyperplanes
    float* h_points = generate_random_points(); // host points
    const unsigned long long int h_prime = 0x9e3779b97f4a7c15ULL;
    /* POINTERS TO HOST */

    /* POINTERS TO DEVICE */
    int **d_buckets; // buckets containing the indexes of the points
    unsigned long long int *d_signatures;              // signatures of each point
    unsigned int *d_bucket_size;             // currently used size of each bucket
    //float* d_hyperplanes;           // points to the random hyperplanes on device
    float *d_points;                // points stored on the device
    /* POINTERS TO DEVICE*/

    /* ALLOCATE MEMORY ON DEVICE */
    cudaFree(0);

    // upload points to device global memory
    cudaMalloc((void**)&d_points, N * DIMENSIONS * sizeof(float));
    cudaMemcpy(d_points, h_points, N * DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);

    // allocate memory for signatures and bucket size on device global memory
    cudaMalloc(&d_signatures, N * sizeof(unsigned long long int));
    cudaMalloc(&d_bucket_size, N * sizeof(int));
    cudaMemset(&d_bucket_size, 0, N * sizeof(int)); // set all buckets size to 0

    allocateMatrixOnDevice(&d_buckets, N, BUCKET_SIZE);

    // upload hyperplanes to device constant memory
    cudaMemcpyToSymbol(d_hyperplanes, h_hyperplanes, sizeof(float) * DIMENSIONS * N_HYPERPLANES);
    // upload prime to constant memory
    cudaMemcpyToSymbol(prime, &h_prime, sizeof(unsigned long long int));

    int number_of_blocks = 1, threads_per_block = 4;
    int n = 1 << 14;

    add_device<<<number_of_blocks, threads_per_block>>>(d_points, d_buckets, d_signatures, d_bucket_size, n);
    cudaDeviceSynchronize();
    printf("done\n");
}