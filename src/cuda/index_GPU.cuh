#include "index.hpp"
#include "point.hpp"
#include "common.hpp"

using namespace std;

#ifndef INDEX_GPU_HPP
#define INDEX_GPU_HPP

/**
 * @brief LSH GPU index
 * 
 * @param number_of_blocks number of blocks
 * @param threads_per_block threads per block
 */
class Index_GPU {
public:
    unsigned int number_of_blocks;
    unsigned int threads_per_block;
    /**
     * @brief Constructs a new Index
     * @param d Number of dimensions
     * @param nbits Number of random hyperplanes
     * @param points Host points
     * @param number_of_blocks number of blocks
     * @param threads_per_block threads per block 
     */
    Index_GPU(int d, int nbits, vector<Point> &points, int number_of_blocks, int threads_per_block);
    
    /**
     * @brief destructor for the GPU index
     */
    ~Index_GPU();

    /**
     * @brief add n points to index in device
     * 
     * @param points the points to adds
     * @param buckets matrix of buckets
     * @param signatures signature of each point
     * @param bucket_size currently used space in each bucket
     * @param n the number of points to add 
     */
    void add(Point *points, int **buckets, int *signatures, int *bucket_size, int n);
    
    /**
     * @brief search for set of points in the index
     */
    __global__ unsigned int* search(Point* p, int n);

private:
    int d;      // point dimensionality
    int nbits;  // number of random hyperplanes

    static const int BUCKET_SIZE = 8;   // fixes size of each bucket    7

    /* POINTERS TO DEVICE */
    __constant__ float device_hyperplanes[DIMENSIONS * N_HYPERPLANES];     // random hyperplanes 
    int **device_buckets;          // buckets containing the indexes of the points
    int *device_signatures;        // signatures of each point
    int *device_bucket_size;       // currently used size of each bucket
    Point *device_points;   // points stored on the device
    /* POINTERS TO DEVICE*/
    
    /**
     * @brief actually add n points to index in device
     * 
     * @param points the points to adds
     * @param buckets matrix of buckets
     * @param signatures signature of each point
     * @param bucket_size currently used space in each bucket
     * @param n the number of points to add 
     */
    __global__ void add_device(Point *points, int **buckets, int *signatures, int *bucket_size, int n);

    /**
     * @brief hash a signature to determine in which bucket to put the point
     * 
     * Each input bit affects each output bit with about 50% probability. There are no collisions (each input results in a different output). The algorithm is fast except if the CPU doesn't have a built-in integer multiplication unit.
     * 
     * The magic number was calculated using a special multi-threaded test program that ran for many hours, which calculates the avalanche effect (the number of output bits that change if a single input bit is changed; should be nearly 16 on average), independence of output bit changes (output bits should not depend on each other), and the probability of a change in each output bit if any input bit is changed. The calculated values are better than the 32-bit finalizer used by MurmurHash, and nearly as good (not quite) as when using AES. A slight advantage is that the same constant is used twice
     * @param signature is the signature to hash
     * @return the hashed signature
     */
    __device__ unsigned int hash_signature(unsigned int signature);

    /**
    * @brief double the allocated space of bucket
    * @param bucket is a pointer to the pointer to the current data
    * @param current_size is the current_size of the bucket
    */
    __device__ void resize(int **bucket, int &current_size);

    /**
     * @brief checks whether n is a power of two
     * @return true if n is a power of two, false otherwise
     */
    __device__ inline bool is_power_of_two(unsigned int n);

    /**
     * @brief calculates the signature of point p
     * @param p is the Point used to calculate the signature
     * @return the signature of point p in form of an unsigned integer
     */
    __device__ unsigned long long int signature_gpu(const Point &p);

    /**
     * @brief generate nbits d-dimensional random hyperplanes
     * @param d point dimensionality
     * @param nbits number of random hyperplanes
     * @return genereated random hyperplanes     * 
     */
    float* generate_random_hyperplanes(int d, int nbits);
};

#endif