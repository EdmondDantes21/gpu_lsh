#include <iostream>
#include <vector>
#include <sys/time.h>
#include <iomanip>
#include "point.hpp"
#include "index.hpp"
#include "index_CPU.hpp"

using namespace std;

#define DIMENSIONS 4
#define N_HYPERPLANES 128
#define N 1 << 19

void print_usage();

int main(int argc, char** argv) {
    if (argc == 1) {
        print_usage();
        return 0;
    }
    
    // SERIAL VERSION ON CPU
    if (argv[1] == string("-serial")) {
        cout << "INSERT\n";
        cout << "n_points" << setw(20) << "time[s]\n";
        struct timeval start, end;
        for (int n_points = 1 << 10; n_points <= N; n_points *= 2) {
            vector<Point> points = generate_points(n_points, DIMENSIONS);
            
            gettimeofday(&start, NULL);
            Index index = Index(DIMENSIONS, N_HYPERPLANES);
            index.add(points);
            gettimeofday(&end, NULL);
            
            long long int time_usec = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
            cout << right << setw(10) << n_points << "\t"
            << fixed << setprecision(6) << setw(20) << time_usec / 1000000.0 << endl;
        }
        
        cout << "SEARCH\n";
        cout << "n_points" << setw(20) << "time[s]\n";

        Index index = Index(DIMENSIONS, N_HYPERPLANES);
        vector<Point> points = generate_points(N, DIMENSIONS);
        index.add(points);

        for (int n_points = 1 << 10; n_points <= N; n_points *=2) {
            vector<Point> points_to_search = generate_points(n_points, DIMENSIONS);
            gettimeofday(&start, NULL);
            
            for (auto& p : points_to_search)
                index.search(p);
            
            gettimeofday(&end, NULL);

            long long int time_usec = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
            cout << right << setw(10) << n_points << "\t"
            << fixed << setprecision(6) << setw(20) << time_usec / 1000000.0 << endl;
        }

    }

    // PARALLEL VERSION ON CPU
    if (argc == 3 && argv[1] == string("-openmp")) {
        int threads = atoi(argv[2]);
        cout << "INSERT\n";
        cout << "n_points" << setw(15) << "time[s]\n";
        struct timeval start, end;
        for (int n_points = 1 << 10; n_points <= N; n_points *= 2) {
            vector<Point> points = generate_points(n_points, DIMENSIONS);
            
            gettimeofday(&start, NULL);
            Index_CPU index = Index_CPU(DIMENSIONS, N_HYPERPLANES, threads);
            index.add(points);
            gettimeofday(&end, NULL);

            long long int time_usec = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
            cout << right << setw(10) << n_points << "\t"
            << fixed << setprecision(6) << setw(20) << time_usec / 1000000.0 << endl;
        }

        cout << "SEARCH\n";
        cout << "n_points" << setw(15) << "time[s]\n";

        Index_CPU index = Index_CPU(DIMENSIONS, N_HYPERPLANES, threads);
        vector<Point> points = generate_points(N, DIMENSIONS);
        index.add(points);

        for (int n_points = 1 << 10; n_points <= N; n_points *=2) {
            vector<Point> points_to_search = generate_points(n_points, DIMENSIONS);

            gettimeofday(&start, NULL);
            vector<optional<Point>> search_result = index.search(points_to_search);
            gettimeofday(&end, NULL);

            long long int time_usec = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
            cout << right << setw(10) << n_points << "\t"
            << fixed << setprecision(6) << setw(20) << time_usec / 1000000.0 << endl;
        }
    }
    return 0;

    // TODO: PARALLEL VERSION ON GPU
}


void print_usage() {
    cout << "Usage: ./lsh [OPTION] [THREADS]\n";
    
    cout << "OPTION:\n";
    cout << "\t -serial \t serial CPU version\n";
    cout << "\t -openmp \t Parallel OpenMP implementation\n\n";

    cout << "THREADS: number of threads to spawn in the openmp option\n";
}