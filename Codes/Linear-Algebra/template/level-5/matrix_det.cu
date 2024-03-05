// determinant_without_library.cu
#include <cuda_runtime.h>

__global__ void determinantKernel(float *input, int n, float *det) {
    extern __shared__ float sharedData[];

    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int gridSize = gridDim.x * blockDim.x;
    int matrixSize = n * n;

    // Shared memory for storing LU decomposition
    for (int k = 0; k < n; k += blockDim.x) {
        int row = idy;
        int col = idx + k;

        if (row < n && col < n) {
            sharedData[row * blockDim.x + col - k] = input[row * n + col];
        }
    }

    __syncthreads();

    float determinant = 1.0f;

    for (int k = 0; k < n - 1; ++k) {
        int pivotRow = k;

        for (int i = k + 1; i < n; ++i) {
            if (abs(sharedData[i * blockDim.x + k]) > abs(sharedData[pivotRow * blockDim.x + k])) {
                pivotRow = i;
            }
        }

        if (sharedData[pivotRow * blockDim.x + k] == 0) {
            determinant = 0.0f;
            break;
        }

        if (pivotRow != k) {
            determinant *= -1.0f; // Swap rows, change sign of determinant
            for (int i = 0; i < blockDim.x; ++i) {
                float temp = sharedData[pivotRow * blockDim.x + i];
                sharedData[pivotRow * blockDim.x + i] = sharedData[k * blockDim.x + i];
                sharedData[k * blockDim.x + i] = temp;
            }
        }

        __syncthreads();

        for (int i = k + 1; i < n; ++i) {
            int row = i;
            int col = idx + k;

            if (row < n && col < n) {
                float factor = sharedData[i * blockDim.x + k] / sharedData[k * blockDim.x + k];
                sharedData[i * blockDim.x + col] -= factor * sharedData[k * blockDim.x + col];
            }
        }

        __syncthreads();
    }

    // Compute the determinant from the upper triangular matrix
    if (idx == 0 && idy == 0) {
        determinant *= sharedData[0];
        for (int i = 1; i < n; ++i) {
            determinant *= sharedData[i * blockDim.x + i];
        }

        *det = determinant;
    }
}
