#include "index.hpp"
#include "point.hpp"

using namespace std;

#ifndef INDEX_GPU_HPP
#define INDEX_GPU_HPP

class Index_GPU : public Index::Index {
public:
    Index_GPU();
    ~Index_GPU();
};

#endif