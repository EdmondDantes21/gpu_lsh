#include <iostream>
#include <omp.h>
#include <random>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <iomanip>
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
 * @brief double the bucket capacity
 * @param bucket is a pointer to the bucket
 * @param current_size is the current size of the bucket
 */
__device__ void resize(int **bucket, unsigned int current_size) {
    // allocate new bucket with twice the size as the old one (requires compute capability of 3.0 or above)
    int *new_bucket = (int*) malloc(current_size * 2 * sizeof(int));

    // copy data from old bucket to new bucket
    for(int i = 0; i < current_size; i++)
        new_bucket[i] = (*bucket)[i];
    
    free(*bucket);   // free old bucket memory
    *bucket = new_bucket; // bucket now points to the new bucket data
}

/**
 * @brief check whether n is a power of two on CUDA device
 * @param n the number to test
 * @return true when n is a power of two, false otherwise
 */
__device__ inline bool is_power_of_two(unsigned int n) {
    return (n & (n - 1)) == 0;
}

/**
 * @brief calculate the signature of a point
 * @param points pointer to the array of all points stored in device global memory
 * @param point_number the index of the point to calculate the signature of
 */
__device__ unsigned long long int signature_gpu(float* points, int point_number) {
    unsigned long long int sig = 0;
    unsigned int exponent = 0;
    float base = 0.0;

    for (int i = 0; i < N_HYPERPLANES; i++) {
        base = 0.0;
        #pragma unroll // when DIMENSIONS IS SMALL
        for (int j = 0; j < DIMENSIONS; j++)
            base = __fmaf_rn(points[point_number + j], d_hyperplanes[DIMENSIONS * i + j], base);

        if (base >= 0.0) 
            sig += (1ULL << exponent);

        exponent++;
    }   
    return sig;
}

/**
 * @brief add points to buckets
 * @param points the vector of points to add
 * @param buckets matrix of n buckets
 * @param signatures array of signature, one for each point (uninitialized)
 * @param curr_buket_used array of integers which indicate, for each bucket, how much is already occupied 
 * @param n the number of points (and buckets)
 * @note the number of buckets is the same as the number of buckets because that is what the cpp implementation of hash-based containers suggests
 */
__global__ void add_device(float *points, int **buckets, unsigned long long int *signatures, unsigned int *curr_bucket_used, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_count = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += thread_count) {
        signatures[i] = signature_gpu(points, i);  // calculate point signature, i.e. whether it is left/right of each hyperplane
        unsigned int bucket_index = (hash_signature(signatures[i]) % (n / thread_count)) + n / thread_count * tid ; // bucket index indicates in which bucket to put this point. Note that we store the index of the point rather than the point itself
        
        if (curr_bucket_used[bucket_index] >= BUCKET_SIZE && is_power_of_two(curr_bucket_used[bucket_index])) { // each time a bucket is full we double its capacity
            resize(&buckets[bucket_index], curr_bucket_used[bucket_index]);
        }

        buckets[bucket_index][curr_bucket_used[bucket_index]] = i; // put the index of the i-th point in the right bucket
        curr_bucket_used[bucket_index]++;
    }
}

/**
 * @brief search n points in the index
 * @param points is the set of points to search
 * @param buckets is the bucket indexes
 * @param buckets_size is the size of each
 * @param result are the resulting indexes
 * @param n the number of points to search
 */
__global__ void search(float *points, int **buckets, unsigned int *bucket_size, unsigned int* result, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_count = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += thread_count) {
        unsigned long long int signature = signature_gpu(points, i);
        unsigned int hamming_zero_bucket = hash_signature(signature);

        if (bucket_size[hamming_zero_bucket] != 0)
            result[i] = buckets[hamming_zero_bucket][0];
    }
}
/**
 * @brief generate nbits d-dimensional hyperplanes
 * @param d dimensionality of the hyperplanes
 * @param nbits
 */
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

/**
 * @brief generate N random points
 * @param n number of points to generate
 * @param dimensions dimensionality of the points to generate
 * @return the randomly generated points
 */
float* generate_random_points(int n, int dimensions) {
    float * points = new float[n * dimensions];
    
    srand(time(NULL));
    const float lower_bound = -1000.0;
    const float upper_bound = 1000.0;
    
    #pragma omp parallel num_threads(4)
    {
        #pragma omp parallel for
        for (int i = 0; i < n * dimensions; i++) {
            points[i] = lower_bound + (upper_bound - lower_bound) * ((float) rand() / RAND_MAX);
        }
    }
    return points;
}

/**
 * @brief allocate an uninitilized matrix on device
 * @param d_matrix pointer to matrix on device
 * @param rows number of rows
 * @param cols number of colums
 */
void allocateMatrixOnDevice(int*** d_matrix, int rows, int cols) {
    int** h_row_ptrs = new int*[rows]; // Host array of pointers

    // Allocate device memory for the array of pointers
    int** d_row_ptrs;
    CHECK_CUDA(cudaMalloc((void**)&d_row_ptrs, rows * sizeof(int*)));

    // Allocate memory for each row on the device
    for (int i = 0; i < rows; ++i) {
        int* d_row;
        CHECK_CUDA(cudaMalloc((void**)&d_row, cols * sizeof(int)));
        h_row_ptrs[i] = d_row; // Store row pointer in host array
    }

    // Copy the row pointers from host to device
    CHECK_CUDA(cudaMemcpy(d_row_ptrs, h_row_ptrs, rows * sizeof(int*), cudaMemcpyHostToDevice));

    // Copy the device row pointers back to the device
    for (int i = 0; i < rows; ++i) {
        CHECK_CUDA(cudaMemcpy(&(d_row_ptrs[i]), &h_row_ptrs[i], sizeof(int*), cudaMemcpyHostToDevice));
    }

    *d_matrix = d_row_ptrs; // Assign the allocated device matrix
    delete[] h_row_ptrs; // Free host row pointers
}

int main(int argc, char** argv) {
    /* PARAMETERS */
    int n = atoi(argv[1]);                  // number of points
    int number_of_blocks = atoi(argv[2]);   // number of CUDA blocks
    int threads_per_block = atoi(argv[3]);  // threads per CUDA block
    bool add = atoi(argv[4]);               // if true, benchmark add, otherwise benchmark insert

    /* POINTERS TO HOST */
    float *h_hyperplanes = generate_random_hyperplanes(DIMENSIONS, N_HYPERPLANES); // host random hyperplanes
    float* h_points = generate_random_points(n, DIMENSIONS); // host points
    const unsigned long long int h_prime = 0x9e3779b97f4a7c15ULL;
    /* POINTERS TO HOST */

    /* POINTERS TO DEVICE */
    int **d_buckets;                        // buckets containing the indexes of the points
    unsigned long long int *d_signatures;   // signatures of each point
    unsigned int *d_bucket_size;            // currently used size of each bucket
    float *d_points;                        // points stored on the device
    /* POINTERS TO DEVICE*/

    /* STARTING EXECUTION ON CUDA DEVICE */
    struct timeval start, end;
    gettimeofday(&start, NULL);

    struct timeval mem_transfer_start, mem_transfer_end;
    gettimeofday(&mem_transfer_start, NULL);

    /* ALLOCATE MEMORY ON DEVICE */

    // Upload points to device global memory (asynchronously)
    CHECK_CUDA(cudaMalloc((void**)&d_points, n * DIMENSIONS * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_points, h_points, n * DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate memory for signatures and bucket size on device global memory (asynchronously)
    CHECK_CUDA(cudaMalloc(&d_signatures, n * sizeof(unsigned long long int)));
    CHECK_CUDA(cudaMalloc(&d_bucket_size, n * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_bucket_size, 0, n * sizeof(int)));  // Set all bucket sizes to 0 asynchronously

    // Upload hyperplanes and prime constant to device constant memory asynchronously
    CHECK_CUDA(cudaMemcpyToSymbol(d_hyperplanes, h_hyperplanes, sizeof(float) * DIMENSIONS * N_HYPERPLANES, 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(prime, &h_prime, sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice));

    allocateMatrixOnDevice(&d_buckets, n, BUCKET_SIZE);

    // increase heap size to allow for dinamic allocation
    size_t heapSize = 1024 * 1024 * 1024; // 1 GB
    CHECK_CUDA(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));

    gettimeofday(&mem_transfer_end, NULL);
    
    add_device<<<number_of_blocks, threads_per_block, 0>>>(d_points, d_buckets, d_signatures, d_bucket_size, n);
    cudaDeviceSynchronize();
    
    gettimeofday(&end, NULL);
    long long int time_usec = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
    long long int time_usec_mem = ((mem_transfer_end.tv_sec * 1000000 + mem_transfer_end.tv_usec) - (mem_transfer_start.tv_sec * 1000000 + mem_transfer_start.tv_usec));
    cout << n << setw(20) << time_usec / 1000000.0 << "(" <<  float(time_usec_mem) / float(time_usec) * 100.0 << "%)" << endl;
    
    // Reset device
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}