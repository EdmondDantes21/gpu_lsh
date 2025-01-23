# Usage

## OPENMP Version
open the `src/openmp/` folder and run the command `make`. 

If you want to do it manually:
compile with `g++ -fopenmp lsh.cpp point.cpp index.cpp index_CPU.cpp`

execute with `./a.out [OPTION] [THREADS]`

`[OPTION]`: 

-`serial`: serial version without OPENMP

-`parallel`: parallel version

`[THREADS]`:

number of threads to spawn when using OPENMP.

## CUDA Version

run the command `make`. If you want to do it manually:

open the `src/cuda/` folder

compile with `nvcc -rdc=true -Xcompiler -fopenmp marzola.cu`

execute with `./a.out [n] [blocks] [threads per block] [add]`

`n`: how many points to create
`blocks`: how many blocks to create
`threads per block`: how many threads per block to create
`add`: 1 if you want to perform the add operation, 0 to also perform the search operation.

Note that for now the points used for search and add are randonmly generated usign the `generate_points` function. If you want to use your own points, customize the aforementioned method.
