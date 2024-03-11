#include <iostream>
#include <vector>
#include "point.hpp"
#include "index.hpp"

using namespace std;

#define N_POINTS 128
#define DIMENSIONS 4
#define N_HYPERPLANES 128

int main() {
    vector<Point> points = generate_points(N_POINTS, DIMENSIONS);
    Index index = Index(DIMENSIONS, N_HYPERPLANES);
}