#include "index.hpp"
#include "point.hpp"

using namespace std;

#ifndef INDEX_CPU_HPP
#define INDEX_CPU_HPP

/**
 * @brief multithreaded LSH Index 
 * 
 * @param n_threads Is the number of threads to spawn
*/
class Index_CPU : public Index::Index {
public:
    Index_CPU();
    ~Index_CPU();
    Index_CPU(int, int, int);
    void add(vector<Point>&);
    optional<Point> search(Point&);
private:
    int n_threads;
    void add(vector<Point>&, int, int);
    void merge(Index_CPU&, Index_CPU&);
};

#endif