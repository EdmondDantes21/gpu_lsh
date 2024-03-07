#include "point.hpp"
#include <vector>
#include <math.h>

using namespace std;

Point::Point(vector<double> v) {
    this->d = v.size();
    this->v = v;
}

Point::Point() {
    this->d = 0;
    this->v = vector<double>(0);
}

Point::~Point() {}

ostream &operator<<(ostream &os, const Point &p) {
    os << "{ ";
    for (auto &el : p.v)
        os << el << " ";
    os << "}";
    return os;
}

/**
 * @brief Generate n d-dimensional points uniformally distributed across all dimensions
 *
 * @param n number of points to generate
 * @param d number of dimensions
 * @return points generated
 */
vector<Point> generate_points(int n, int d) {
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
double dist(const Point& p1, const Point& p2) {
    double sum = 0.0;
    for (int i = 0; i < p1.d; i++)
        sum += pow((p1.v[i] - p2.v[i]), 2);
    return sqrt(sum);
}