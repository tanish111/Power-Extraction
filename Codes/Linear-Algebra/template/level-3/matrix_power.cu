// element_power.cu
#include <cuda_runtime.h>
#include <cmath>

__global__ void elementPowerKernel(float *input, float *output, int rows, int cols, int exponent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        output[index] = pow(input[index], exponent);
    }
}
