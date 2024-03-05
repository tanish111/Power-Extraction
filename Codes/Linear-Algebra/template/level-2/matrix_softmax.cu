// softmax.cu
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmaxKernel(float *input, float *output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows) {
        // Find the maximum value in the input row
        float max_val = input[idx * cols];
        for (int i = 1; i < cols; ++i) {
            if (input[idx * cols + i] > max_val) {
                max_val = input[idx * cols + i];
            }
        }

        // Subtract the maximum value for numerical stability
        float sum_exp = 0.0f;
        for (int i = 0; i < cols; ++i) {
            int index = idx * cols + i;
            output[index] = expf(input[index] - max_val);
            sum_exp += output[index];
        }

        // Normalize by the sum of exponentials to obtain the softmax result
        for (int i = 0; i < cols; ++i) {
            int index = idx * cols + i;
            output[index] /= sum_exp;
        }
    }
}
