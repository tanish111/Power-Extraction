#include <iostream>
#include <cstdlib>
#include <ctime>
#include "matrix_det.cu"
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
    int c=0;
    for (int i = 0; i < n * n; ++i)
    {
        if(i==n*c+c) 
        {h_matrix[i]=1;
        c++;
        }
        else h_matrix[i]=0;
    }

    // // Print the original matrix
    // std::cout << "Original Matrix:" << std::endl;
    // printMatrix(h_matrix, n, n);

    // Allocate device memory for the matrix
    float *d_matrix;
    cudaMalloc((void **)&d_matrix, sizeof(float) * n * n);

    // Copy the host matrix to the device
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * n * n, cudaMemcpyHostToDevice);

// Allocate device memory for the result determinant
float h_result_determinant;
float *d_result_determinant;
cudaMalloc((void **)&d_result_determinant, sizeof(float));

// Define grid and block dimensions for determinantKernel
    dim3 BlockSize(B_S, B_S);
    dim3 GridSize((n + BlockSize.x - 1) / BlockSize.x,
                  (n + BlockSize.y - 1) / BlockSize.y);

// Launch the determinantKernel to compute the determinant
GPUDevice g1 = GPUDevice(0,xstr (KERNEL),GridSize.x*GridSize.y*GridSize.z,BlockSize.x*BlockSize.y*BlockSize.z);
g1.startReading();
for(int i=0;i<10;i++){
KERNEL<<<GridSize, BlockSize, sizeof(float) * BlockSize.x * BlockSize.y>>>(d_matrix, n, d_result_determinant);
    cudaDeviceSynchronize();
    }
    // Wait for all threads to finish
    g1.stopReading();
// Wait for all threads to finish
cudaDeviceSynchronize();

// Copy the result determinant back to the host
cudaMemcpy(&h_result_determinant, d_result_determinant, sizeof(float), cudaMemcpyDeviceToHost);

// Print the resultant determinant
std::cout << "Determinant: " << h_result_determinant << std::endl;

// Free memory on the device for the result determinant
cudaFree(d_result_determinant);

    // // Print the resultant matrix
    // printMatrix(h_matrix, n, n);

    // Free memory on the host and device
    delete[] h_matrix;
    cudaFree(d_matrix);

    return 0;
}
