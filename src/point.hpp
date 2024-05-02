#include <vector>
#include <iostream>

using namespace std;

#ifndef POINT_HPP
#define POINT_HPP
/**
 * @brief A d-dimensional point
 * @param v d-dimensional vector with the values of each dimension
 */
class Point {
public:
    vector<double> v;

    Point();
    Point(vector<double>);
    ~Point();
    friend ostream& operator<<(ostream& os, const Point& p);
    bool operator==(const Point& otherPoint) const;
    
    struct HashFunction {
        size_t operator()(const Point& point) const {
            size_t seed = point.v.size();
            for (auto & d : point.v) 
                seed ^= hash<double>()(d) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
};

vector<Point> generate_points(int, int);
double dist(const Point&, const Point&);
#endif