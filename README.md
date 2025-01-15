# Usage

## OPENMP Version
run the command `make`. If you want to do it manually:
compile with `g++ -fopenmp lsh.cpp point.cpp index.cpp index_CPU.cpp`

execute with `./a.out [OPTION] [THREADS]`

`[OPTION]`: 

-`serial`: serial version without OPENMP

-`parallel`: parallel version

`[THREADS]`:

number of threads to spawn when using OPENMP.

## CUDA Version

run the command `make`. If you want to do it manually:

compile with `! nvcc -rdc=true -Xcompiler -fopenmp marzola.cu`

execute with `./a.out`

Note that for now the points used for search and add are randonmly generated usign the `generate_points` function. If you want to use your own points, customize the aforementioned method.
