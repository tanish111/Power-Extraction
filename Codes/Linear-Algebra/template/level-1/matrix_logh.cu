// elementwise_log.cu
#include <cuda_runtime.h>
#include <cmath>

__global__ void elementwiseLogKernel(float *input, float *output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        output[index] = logf(input[index]);
    }
}
