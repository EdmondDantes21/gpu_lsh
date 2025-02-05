#include <vector>
#include <math.h>
#include "point.hpp"

using namespace std;

__host__ __device__ Point::Point(vector<double> v) {
    this->v = v;
}

__host__ __device__ Point::Point() {
    this->v = vector<double>(0);
}

__host__ __device__ Point::~Point() {}

__host__ __device__ ostream &operator<<(ostream &os, const Point &p) {
    os << "{ ";
    for (auto &el : p.v)
        os << el << " ";
    os << "}";
    return os;
}

/**
 * @brief Overload of the operator == used for making the class Point hashable
 * 
 * @param other is the point to compare
 * @return true if the two points are equal
 * @return false otherwise
 */
__host__ __device__ bool Point::operator==(const Point& other) const {
    return this->v == other.v;
}

/**
 * @brief Generate n d-dimensional points uniformally distributed across all dimensions
 *
 * @param n number of points to generate
 * @param d number of dimensions
 * @return points generated
 */
__host__ __device__ vector<Point> generate_points(int n, int d) {
    vector<Point> points(n);

    srand(time(NULL));
    const long max_rand = 100000000L;
    double lower_bound = -1000000.0;
    double upper_bound = 1000000.0;

    for (int i = 0; i < n; i++) {
        vector<double> v(d);
        for (int j = 0; j < d; j++)
            v[j] = lower_bound + (upper_bound - lower_bound) * (rand() % max_rand) / max_rand;
        points[i] = Point(v);
    }
    return points;
}

/**
 * @brief Calculate the distance between two Points
 * 
 * @param p1 The first point
 * @param p2 The second point
 * @return the distance between p1 and p2
 */
__host__ __device__ double dist(const Point& p1, const Point& p2) {
    double sum = 0.0;
    for (int i = 0; i < p1.v.size(); i++)
        sum += pow((p1.v[i] - p2.v[i]), 2);
    return sqrt(sum);
}