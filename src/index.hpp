#include <unordered_map>
#include <unordered_set>
#include <set>
#include "point.hpp"

using namespace std;

#ifndef INDEX_HPP
#define INDEX_HPP
/**
 * @brief LSH index
 * 
 * @param d Dimensionality of the points
 * @param nbits Number of random hyperplanes
 * @param map The index it self. The key is a nbits dimensional boolean vector and the value is a set of Points
 * @param hyperplanes The random hyperplanes 
 */
class Index {
public:
    Index();
    Index(int, int);
    ~Index();
    void add(Point&);
    void search(Point&);
    struct VectorBoolHash {};
    struct VectorBoolEqual {};  
private:
    int d;
    int nbits;
    unordered_map<vector<bool>, unordered_set<Point, Point::HashFunction>, VectorBoolHash, VectorBoolEqual> map;
    vector<vector<double>> hyperplanes;
    vector<bool> signature(const Point&);
};

vector<vector<double>> generate_hyperplanes(int, int);
#endif