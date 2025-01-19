#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "point_gpu.hpp"
#include "common.hpp"

using namespace std;

__constant__ float d_hyperplanes[DIMENSIONS * N_HYPERPLANES];     // random hyperplanes 

__device__ unsigned int hash_signature(unsigned int signature) {
    signature = ((signature >> 16) ^ signature) * 0x45d9f3b;
    signature = ((signature >> 16) ^ signature) * 0x45d9f3b;
    signature = (signature >> 16) ^ signature;
    return signature;
}

__device__ void resize(int** bucket, int& current_size) {
    int* new_bucket;
    cudaMalloc(&new_bucket, current_size * 2 * sizeof(int)); // allocate a new bucket on GPU of new_size ints
    for (int i = 0; i < current_size; i++) // copy data from old bucket to new bucket
        new_bucket[i] = bucket[0][i];
    // cudaFree(bucket[0]);   // free old bucket memory
    *bucket = new_bucket; // bucket now points to the new bucket data
    current_size *= 2; // new size is double the current size
}

__device__ inline bool is_power_of_two(unsigned int n) {
    return (n & (n - 1)) == 0;
}

__device__ unsigned long long int signature_gpu(const Point &p) {
    unsigned long long int sig = 0;
    unsigned int exponent = 0;
    int base = 0.0;

    for (int i = 0; i < N_HYPERPLANES; i++) {
        for (int j = 0; j < DIMENSIONS; j++)
            base += p.v[j] * d_hyperplanes[N_HYPERPLANES * i + j];
        
        sig += pow(base, exponent);
        base = 0.0;
        exponent *= 2;
    }
    return sig;
}

__global__ void add_device(Point* points, int **buckets, int* signatures, int* bucket_size, int n) {
    for (int i = 0; i < n; i++) {
        signatures[i] = signature_gpu(points[i]);   // calculate point signature, i.e. whether it is left/right of each hyperplane
        unsigned int bucket_index = hash_signature(signatures[i]) % n;  // bucket index indicates in which bucket to put this point. Note that we store the index of the point rather than the point itself
        if (bucket_size[bucket_index] >= BUCKET_SIZE && is_power_of_two(bucket_size[bucket_index])) // each time a bucket is full we double its capacity
            resize(&buckets[bucket_index], bucket_size[bucket_index]);
        buckets[bucket_index][bucket_size[bucket_index]] = i; // put the index of the i-th point in the right bucket
        bucket_size[bucket_index]++;
    }

    //print bucket content
    for (int i = 0; i < n; i++) {
        printf("BUCKET[%d]->", i);
        for (int j = 0; j < bucket_size[i]; j++)
            printf("%d ", buckets[i][j]);
        printf("\n");
    }
}

__global__ void search(Point* points, int n) {

}

float* generate_random_hyperplanes(int d, int nbits) {
    float * hyperplanes = new float[d * nbits];

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < nbits; i++) 
        for (int j = 0; j < d; j++) 
            hyperplanes[i * nbits + j] = dis(gen);
    return hyperplanes;
}

/**
 * @brief Generate n d-dimensional points uniformally distributed across all dimensions
 *
 * @param n number of points to generate
 * @param d number of dimensions
 * @return points generated
 */
vector<Point> generate_random_points(int n, int d) {
    vector<Point> points(n);

    srand(time(NULL));
    const long max_rand = 100000000L;
    float lower_bound = -1000000.0;
    float upper_bound = 1000000.0;

    for (int i = 0; i < n; i++) {
        float* v = new float[d];
        for (int j = 0; j < d; j++)
            v[j] = lower_bound + (upper_bound - lower_bound) * (rand() % max_rand) / max_rand;
        points[i] = Point(v, d);
    }
    return points;
}


int main() {
    /* POINTERS TO HOST */
    float* h_hyperplanes = generate_random_hyperplanes(DIMENSIONS, N_HYPERPLANES);  // host random hyperplanes
    vector<Point> h_points = generate_random_points(DIMENSIONS, N);                 // host points
    /* POINTERS TO HOST */
    /* POINTERS TO DEVICE */
    int **d_buckets;          // buckets containing the indexes of the points
    int *d_signatures;        // signatures of each point
    int *d_bucket_size;       // currently used size of each bucket
    Point *d_points;          // points stored on the device
    /* POINTERS TO DEVICE*/

    /* ALLOCATE MEMORY ON DEVICE */
    cudaFree(0);

    // upload points to device global memory
    cudaMalloc(&d_points, h_points.size() * sizeof(Point));
    cudaMemcpy(d_points, h_points.data(), h_points.size() * sizeof(Point), cudaMemcpyHostToDevice);

    // allocate memory for signatures and bucket size on device global memory
    cudaMalloc(&d_signatures, h_points.size() * sizeof(int)); 
    cudaMalloc(&d_bucket_size, h_points.size() * sizeof(int)); 
    cudaMemset(&d_bucket_size, 0, h_points.size() * sizeof(int));   // set all buckets size to 0

    // allocate memory for the buckets on the device global memory
    cudaMalloc(&d_buckets, h_points.size() * sizeof(int)); 
    for (int i = 0; i < h_points.size(); i++) 
        cudaMalloc(&d_buckets[i], BUCKET_SIZE * sizeof(int));

    // upload calculated random hyperplanes to device
    cudaMemcpyToSymbol(d_hyperplanes, h_hyperplanes, DIMENSIONS * N_HYPERPLANES * sizeof(float), cudaMemcpyHostToDevice);

    int number_of_blocks = 8, threads_per_block = 64;

    add_device<<<number_of_blocks, threads_per_block>>>(d_points, d_buckets, d_signatures, d_bucket_size, N);
    cudaDeviceSynchronize();

}
