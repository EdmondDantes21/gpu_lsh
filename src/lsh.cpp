#include <iostream>
#include <vector>
#include "point.hpp"
#include "index.hpp"

using namespace std;

#define N_POINTS 10
#define DIMENSIONS 2
#define N_HYPERPLANES 4

int main() {
    vector<Point> points = generate_points(N_POINTS, DIMENSIONS);
    Index index = Index(DIMENSIONS, N_HYPERPLANES);

    // index.add(points);
    for (auto& point : points)
        index.add(point);

    index.print();
}