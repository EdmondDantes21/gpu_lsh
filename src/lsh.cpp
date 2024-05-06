#include <iostream>
#include <vector>
#include <sys/time.h>
#include "point.hpp"
#include "index.hpp"
#include "index_CPU.hpp"

using namespace std;

#define DIMENSIONS 4
#define N_HYPERPLANES 128

void print_usage();

int main(int argc, char** argv) {
    if (argc == 1) {
        print_usage();
        return 0;
    }
    
    // SERIAL VERSION ON CPU
    if (argv[1] == string("-serial")) {
        cout << "n_points \t time_usec \t time_sec\n";
        struct timeval start, end;
        for (int n_points = 1 << 10; n_points <= 1 << 20; n_points *= 2) {
            gettimeofday(&start, NULL);
            
            vector<Point> points = generate_points(n_points, DIMENSIONS);
            Index index = Index(DIMENSIONS, N_HYPERPLANES);
            index.add(points);
            
            gettimeofday(&end, NULL);
            long long int time_usec = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
            cout << n_points << " \t " << time_usec << " \t " << time_usec / 1000000.0 << endl;
        }
    }

    // PARALLEL VERSION ON CPU
    if (argc == 3 && argv[1] == string("-openmp")) {
        int threads = atoi(argv[2]);
        cout << "n_points \t time_usec \t time_sec\n";
        struct timeval start, end;
        for (int n_points = 1 << 10; n_points <= 1 << 20; n_points *= 2) {
            gettimeofday(&start, NULL);
            
            vector<Point> points = generate_points(n_points, DIMENSIONS);
            Index_CPU index = Index_CPU(DIMENSIONS, N_HYPERPLANES, threads);
            index.add(points);

            gettimeofday(&end, NULL);
            long long int time_usec = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
            cout << n_points << " \t " << time_usec << " \t " << time_usec / 1000000.0 << endl;
        }
    }
    return 0;
}


void print_usage() {
    cout << "Usage: ./lsh [OPTION] [THREADS]\n";
    
    cout << "OPTION:\n";
    cout << "\t -procedural \t Procedural CPU version\n";
    cout << "\t -openmp \t Parallel OpenMP implementation\n\n";

    cout << "THREADS: number of threads to spawn in the openmp option\n";
}