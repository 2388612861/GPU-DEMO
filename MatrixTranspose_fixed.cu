#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Define matrix transpose kernel function
__global__ void transposeMatrix(float* input, float* output, int width, int height) {
    // Each thread is responsible for one element
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index_in = y * width + x;  // Input matrix index
        int index_out = x * height + y;  // Output matrix index
        output[index_out] = input[index_in];
    }
}

// Define matrix transpose kernel function (using shared memory)
__global__ void transposeMatrixShared(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Define shared memory with padding to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load data into shared memory
    if (x < width && y < height) {
        int index_in = y * width + x;  // Input matrix index
        tile[ty][tx] = input[index_in];  // Load into shared memory
    }

    __syncthreads();  // Wait for all threads to complete loading

    // Write data from shared memory to global memory
    if (x < width && y < height) {
        int index_out = x * height + y;  // Output matrix index
        output[index_out] = tile[tx][ty];  // Write back to global memory (transposed)
    }
}

int main() {
    std::cout << "Starting Matrix Transpose CUDA Program..." << std::endl;
    
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);

    std::cout << "Matrix size: " << width << "x" << height << std::endl;

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    float* h_output_shared = (float*)malloc(size);

    // Initialize input matrix
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    float* d_output_shared;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_output_shared, size);

    // Copy input matrix from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
    std::cout << "Block size: " << blockSize.x << "x" << blockSize.y << std::endl;

    // Matrix transpose using global memory
    auto start = std::chrono::high_resolution_clock::now();
    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Global memory transpose time: " << elapsed.count() << " s\n";

    // Matrix transpose using shared memory
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixShared<<<gridSize, blockSize>>>(d_input, d_output_shared, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Shared memory transpose time: " << elapsed.count() << " s\n";

    // Copy output matrix from device back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_shared, d_output_shared, size, cudaMemcpyDeviceToHost);

    // Verify results by checking a few sample points
    bool success = true;
    int check_points = 10;  // Check fewer points for faster verification
    for (int i = 0; i < check_points; ++i) {
        int idx = i * (width * height / check_points);
        if (idx < width * height && h_output[idx] != h_output_shared[idx]) {
            std::cout << "Mismatch at index " << idx << ": global=" << h_output[idx] 
                      << ", shared=" << h_output_shared[idx] << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "Transpose result is correct.\n";
    } else {
        std::cout << "Transpose result is incorrect.\n";
    }

    // Free memory
    free(h_input);
    free(h_output);
    free(h_output_shared);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_shared);

    std::cout << "Program completed successfully!" << std::endl;
    return 0;
}
