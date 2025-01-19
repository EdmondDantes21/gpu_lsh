#include <cuda_runtime.h>
#include <array>
#include <random>
#include "index.hpp"
#include "index_GPU.cuh"
#include "point_gpu.hpp"


__device__ unsigned int hash_signature(unsigned int signature) {
    signature = ((signature >> 16) ^ signature) * 0x45d9f3b;
    signature = ((signature >> 16) ^ signature) * 0x45d9f3b;
    signature = (signature >> 16) ^ signature;
    return signature;
}

__device__ void resize(int** bucket, int& current_size) {
    int* new_bucket;
    cudaMalloc(&new_bucket, current_size * 2 * sizeof(int)); // allocate a new bucket on GPU of new_size ints
    cudaMemcpy(new_bucket, &bucket, current_size * sizeof(int), cudaMemcpyDeviceToDevice); // copy data from old bucket to new bucket
    cudaFree(bucket[0]);   // free old bucket memory
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

    for (int i = 0; i < nbits; i++) {
        for (int j = 0; j < d; j++)
            base += p.v[j] * device_hyperplanes[nbits * i + j];
        
        sig += pow(base, exponent);
        base = 0.0;
        exponent *= 2;
    }
    return sig;
}

void Index_GPU::add(Point* points, int **buckets, int* signatures, int* bucket_size, int n) {
    add_device<<<number_of_blocks, threads_per_block>>>(device_points, buckets, signatures, bucket_size, n);
    cudaDeviceSynchronize();
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
        cout << "BUCKET[ " << i << "] -> ";
        for (int j = 0; j < bucket_size[i]; j++)
            cout << buckets[i][j] << " ";
        cout << endl;
    }
}

__global__ unsigned int* search(Point* points, int n) {

}

float* Index_GPU::generate_random_hyperplanes(int d, int nbits) {
    float * hyperplanes = new float[d * nbits];

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < nbits; i++) 
        for (int j = 0; j < d; j++) 
            hyperplanes[i * nbits + j] = dis(gen);
    return hyperplanes;
}

Index_GPU::Index_GPU(int d, int nbits, vector<Point> &points, int number_of_blocks, int threads_per_block) {
    this->d = d;
    this->nbits = nbits;
    this->number_of_blocks = number_of_blocks;
    this->threads_per_block = threads_per_block;
    float* hyperplanes = generate_random_hyperplanes(d, nbits);
    
    cudaFree(0);

    // upload points to device global memory
    cudaMalloc(&device_points, points.size() * sizeof(Point));
    cudaMemcpy(device_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice);

    // allocate memory for private parameters
    cudaMalloc(&device_signatures, points.size() * sizeof(int)); 
    cudaMalloc(&device_bucket_size, points.size() * sizeof(int)); 
    cudaMemset(&device_bucket_size, 0, points.size() * sizeof(int));   // set all buckets size to 0
    cudaMalloc(&device_buckets, points.size() * sizeof(int)); 
    
    // upload calculated random hyperplanes to device
    cudaMemcpyToSymbol(device_hyperplanes, hyperplanes, d * nbits * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < points.size(); i++) 
        cudaMalloc(&device_buckets[i], BUCKET_SIZE * sizeof(int));
    
    cudaDeviceSynchronize();
}

Index_GPU::~Index_GPU() {
    cudaDeviceReset();
}