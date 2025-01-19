#include <vector>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#ifndef POINT_HPP
#define POINT_HPP
/**
 * @brief A d-dimensional point
 * @param v d-dimensional vector with the values of each dimension
 */
__host__ __device__ class Point {
public:
    vector<double> v;

    __host__ __device__ Point();
    __host__ __device__ Point(vector<double>);
    __host__ __device__ ~Point();
    __host__ __device__ friend ostream& operator<<(ostream& os, const Point& p);
    __host__ __device__ bool operator==(const Point& otherPoint) const;
    
    __host__ __device__ struct HashFunction {
        size_t operator()(const Point& point) const {
            size_t seed = point.v.size();
            for (auto & d : point.v) 
                seed ^= hash<double>()(d) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
};

vector<Point> generate_points(int, int);
__host__ __device__ double dist(const Point&, const Point&);
#endif