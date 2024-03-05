// relu.cu
#include <cuda_runtime.h>

__global__ void reluKernel(float *input, float *output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        output[index] = fmaxf(0.0f, input[index]); // ReLU activation function
    }
}
