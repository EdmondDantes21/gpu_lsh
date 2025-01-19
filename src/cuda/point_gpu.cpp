#include "point_gpu.hpp"

Point::Point(float* v, int d){
    this->v = new float[d];
    for (int i = 0; i < d; i++)
        this->v[i] = v[i];
    this->d = d;
}

Point::Point() {}

Point::~Point() {}