#include <vector>
#include <iostream>

using namespace std;

#ifndef POINT_HPP
#define POINT_HPP
/**
 * @brief A point in 2D space
 *
 * @param d Number of dimensions
 * @param v d-dimensional vector the values of the point
 */
class Point {
public:
    int d;
    vector<double> v;

    Point();
    Point(vector<double>);
    ~Point();
    friend ostream& operator<<(ostream& os, const Point& p);
};

vector<Point> generate_points(int, int);
double dist(const Point&, const Point&);
#endif