#include <iostream>
#include <cstdlib>
#include <ctime>
#include "scalar_add.cu"
#include "scalar_mul.cu"
#include "matrix_power.cu"
#include "GPUDevice.h"
#define xstr(s) str(s)
#define str(s) #s
void printMatrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    // Check for correct number of command line arguments
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    // Parse matrix size from command line argument
    int n = atoi(argv[1]);

    // Set up random seed
    std::srand(std::time(0));

    // Allocate host memory for the matrix
    float *h_matrix = new float[n * n];

    // Generate random matrix with values between 1 and 10
    for (int i = 0; i < n * n; ++i)
    {
        h_matrix[i] = static_cast<float>(std::rand()) / RAND_MAX * 9.0f + 1.0f;
    }

    // Print the original matrix
    // std::cout << "Original Matrix:" << std::endl;
    // printMatrix(h_matrix, n, n);

    // Allocate device memory for the matrix
    float *d_matrix;
    cudaMalloc((void **)&d_matrix, sizeof(float) * n * n);

    // Copy the host matrix to the device
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    // Allocate device memory for the transposed matrix
    float *d_result_matrix;
    cudaMalloc((void **)&d_result_matrix, sizeof(float) * n * n);

    // Define grid and block dimensions for transpose kernel
    dim3 BlockSize(B_S, B_S);
    dim3 GridSize((n + BlockSize.x - 1) / BlockSize.x,
                  (n + BlockSize.y - 1) / BlockSize.y);

    // Launch the kernel
    // transposeKernel<<<transposeGridSize, transposeBlockSize>>>(d_matrix, d_transposed_matrix, n, n);
    // Scalar to be added
    float scalar = 5.0f;

    // Launch the scalarAddKernel
    GPUDevice g1 = GPUDevice(0,xstr (KERNEL),GridSize.x*GridSize.y*GridSize.z,BlockSize.x*BlockSize.y*BlockSize.z);
    g1.startReading();
    for(int i=0;i<10;i++){
    KERNEL<<<GridSize, BlockSize>>>(d_matrix, d_result_matrix, n, n,scalar);
    }
    // Wait for all threads to finish
    cudaDeviceSynchronize();
    g1.stopReading();
    // Copy the transposed matrix back to the host
    cudaMemcpy(h_matrix, d_result_matrix, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

    // Free memory on the host and device for the transposed matrix
    cudaFree(d_result_matrix);

    // Print the resultant matrix
    // printMatrix(h_matrix, n, n);

    // Free memory on the host and device
    delete[] h_matrix;
    cudaFree(d_matrix);

    return 0;
}
