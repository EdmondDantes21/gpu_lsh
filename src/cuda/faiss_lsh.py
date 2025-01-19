import faiss
import time
import numpy as np

# Parameters
dimensions = 4
nbits = 128
npoints = 1048576

# Generate random points
points = np.random.random((npoints, dimensions)).astype(np.float32)

start_time = time.time()

# Create CPU index
index_cpu = faiss.IndexLSH(dimensions, nbits)

# Move index to GPU
gpu_res = faiss.StandardGpuResources()  # Create GPU resources
index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)  # Move index to GPU (0 is for the first GPU)

# Add points to the GPU index
index_gpu.add(points)

end_time = time.time()
time_sec = end_time - start_time
time_usec = time_sec * 1000 * 1000
usec_per_point = time_usec / npoints

print("Execution time = ", time_sec, "seconds")
print("Execution time = ", time_usec, "microseconds")
print("Microseconds per point: ", usec_per_point)
