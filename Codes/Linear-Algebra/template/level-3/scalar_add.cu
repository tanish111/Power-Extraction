// scalar_add_inplace.cu
#include <cuda_runtime.h>

__global__ void scalarAddKernel(float *input, float *output, int rows, int cols, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        output[index] = input[index] + scalar;
    }
}
