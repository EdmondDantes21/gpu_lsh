#include <iostream>
#include <unordered_map>
#include <vector>
#include <bitset>
#include "point.hpp"

using namespace std;

#define N 128
#define D 4

int main() {
    vector<Point> points = generate_points(N, D);

    for (auto &el : points)
        cout << el << endl; 

    cout << "dist between " << points[0] << " and " << points[1] << " = " << dist(points[0], points[1]) << endl;

    unordered_map<bitset<128>, vector<unsigned int>()> index();

    bitset<128> bs0 = 128;
    bitset<128> bs1 = 64;
    bitset<128> bs2 = 32;
    bitset<128> bs3 = 16;
}