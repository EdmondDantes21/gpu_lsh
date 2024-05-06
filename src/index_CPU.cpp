#include <vector>
#include <optional>
#include <stdint.h> // for the log2 function
#include <unistd.h> // For sleep function
#ifdef _OPENMP
#include <omp.h>
#endif
#include "point.hpp"
#include "index.hpp"
#include "index_CPU.hpp"

using namespace std;

/**
 * @brief Calculates the log2 of an 32 bit integer
*/
static inline uint32_t log2(const uint32_t x) {
    uint32_t y;
    asm ( "\tbsr %1, %0\n"
        : "=r"(y)
        : "r" (x)
    );
    return y;
}

/**
 * @brief Construct a new Index with default values: d = 2 and nbits = 32
*/
Index_CPU::Index_CPU() : Index::Index() {}

/**
 * @brief Construct a new Index
 * 
 * @param d Number of dimension
 * @param nbits Number of random hyperplanes
 * @param n_threads Number of threads to spawn
*/
Index_CPU::Index_CPU(int d, int nbits, int n_threads) : Index::Index(d, nbits) {
    this->n_threads = n_threads;
}

Index_CPU::~Index_CPU() {}

/**
 * @brief Add a slice of a vector to the index
 * 
 * @param points The vector to take the slice from
 * @param start Starting index of the slice (included)
 * @param end Ending index of the slice (exluded)
*/
void Index_CPU::add(vector<Point>& points, int start, int end){
    while (start != end)
        add(points[start++]);
}

/**
 * @brief Add a vector of points to the index
*/
void Index_CPU::add(vector<Point>& points) {
    vector<Index_CPU> indexes(n_threads, Index_CPU(this->d, this->nbits, this->n_threads)); // create one empty index per thread

    #pragma omp parallel num_threads(n_threads)
    {
        int thread_id = omp_get_thread_num();
        int new_index = thread_id;
        int jump = 1;
        int iterations = log2(n_threads);
        int slice_size = points.size() / n_threads;
        bool active = true;

        indexes[thread_id].add(points, thread_id * slice_size, thread_id * slice_size * 2); // add points to the local index
        #pragma omp barrier
        
        while (active && iterations != 0) {
            if (active && new_index % 2 == 0) {
                merge(indexes[thread_id], indexes[new_index * 2]);
                new_index /= 2;
            }   
            iterations--;  
            slice_size *= 2;    
            jump *= 2;
            #pragma omp barrier
        }
    }
}

void Index_CPU::merge(Index_CPU& i1, Index_CPU& i2) {

}

optional<Point> Index_CPU::search(Point& p) {
    return nullopt;
}

