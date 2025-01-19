using namespace std;
#include <cuda_runtime.h>

#ifndef POINT_GPU_HPP
#define POINT_GPU_HPP

class Point {
public:
    float* v;  // Pointer to the values array
    int d;     // Dimension of the point
    
    __host__ __device__ Point() : v(nullptr), d(0) {}
    __host__ __device__ Point(float* v, int d) : d(d) {
        // Allocate and copy data
        this->v = new float[d];
        for (int i = 0; i < d; i++) {
            this->v[i] = v[i];
        }
    }

    // Copy constructor (deep copy)
    __host__ __device__ Point(const Point& other) : d(other.d) {
        v = new float[d];
        for (int i = 0; i < d; i++) {
            v[i] = other.v[i];
        }
    }

    // Copy assignment operator (deep copy)
    __host__ __device__ Point& operator=(const Point& other) {
        if (this != &other) { // Prevent self-assignment
            delete[] v;       // Clean up existing resources

            d = other.d;
            v = new float[d];
            for (int i = 0; i < d; i++) {
                v[i] = other.v[i];
            }
        }
        return *this;
    }

    __host__ __device__ ~Point() {
        if (v) delete[] v;
    }
};

#endif