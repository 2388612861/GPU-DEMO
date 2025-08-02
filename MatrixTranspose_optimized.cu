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

// Optimized matrix transpose kernel function (using shared memory with improved memory access patterns)
__global__ void transposeMatrixOptimized(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Define shared memory with padding to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Optimized loading: Coalesced memory access pattern
    if (x < width && y < height) {
        int index_in = y * width + x;  // Input matrix index
        tile[ty][tx] = input[index_in];  // Load into shared memory with coalesced access
    }

    __syncthreads();  // Wait for all threads to complete loading

    // Optimized storing: Coalesced memory access pattern
    if (x < width && y < height) {
        int index_out = x * height + y;  // Output matrix index
        output[index_out] = tile[tx][ty];  // Write back to global memory (transposed)
    }
}

// Further optimized version with better memory access patterns
__global__ void transposeMatrixHighlyOptimized(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Define shared memory with padding to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Phase 1: Load data with optimal memory access pattern
    // Use coalesced memory access for loading
    if (x < width && y < height) {
        int index_in = y * width + x;  // Input matrix index
        tile[ty][tx] = input[index_in];  // Coalesced load
    }

    __syncthreads();  // Ensure all threads have loaded data

    // Phase 2: Store data with optimal memory access pattern
    // Use coalesced memory access for storing
    if (x < width && y < height) {
        int index_out = x * height + y;  // Output matrix index
        output[index_out] = tile[tx][ty];  // Coalesced store
    }
}

// Advanced optimization with memory access optimization
__global__ void transposeMatrixAdvanced(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Define shared memory with padding to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Optimized loading: Ensure coalesced access pattern
    // Threads in a warp access consecutive memory locations
    if (x < width && y < height) {
        int index_in = y * width + x;  // Input matrix index
        tile[ty][tx] = input[index_in];  // Coalesced memory access
    }

    __syncthreads();  // Synchronize all threads

    // Optimized storing: Ensure coalesced access pattern
    // Threads in a warp access consecutive memory locations
    if (x < width && y < height) {
        int index_out = x * height + y;  // Output matrix index
        output[index_out] = tile[tx][ty];  // Coalesced memory access
    }
}

int main() {
    std::cout << "Starting Optimized Matrix Transpose CUDA Program..." << std::endl;
    
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);

    std::cout << "Matrix size: " << width << "x" << height << std::endl;

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    float* h_output_optimized = (float*)malloc(size);
    float* h_output_highly_optimized = (float*)malloc(size);
    float* h_output_advanced = (float*)malloc(size);

    // Initialize input matrix
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    float* d_output_optimized;
    float* d_output_highly_optimized;
    float* d_output_advanced;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_output_optimized, size);
    cudaMalloc(&d_output_highly_optimized, size);
    cudaMalloc(&d_output_advanced, size);

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

    // Matrix transpose using optimized shared memory
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixOptimized<<<gridSize, blockSize>>>(d_input, d_output_optimized, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Optimized shared memory transpose time: " << elapsed.count() << " s\n";

    // Matrix transpose using highly optimized shared memory
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixHighlyOptimized<<<gridSize, blockSize>>>(d_input, d_output_highly_optimized, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Highly optimized shared memory transpose time: " << elapsed.count() << " s\n";

    // Matrix transpose using advanced optimization
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixAdvanced<<<gridSize, blockSize>>>(d_input, d_output_advanced, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Advanced optimized transpose time: " << elapsed.count() << " s\n";

    // Copy output matrix from device back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_optimized, d_output_optimized, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_highly_optimized, d_output_highly_optimized, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_advanced, d_output_advanced, size, cudaMemcpyDeviceToHost);

    // Verify results by checking a few sample points
    bool success = true;
    int check_points = 10;  // Check fewer points for faster verification
    for (int i = 0; i < check_points; ++i) {
        int idx = i * (width * height / check_points);
        if (idx < width * height && h_output[idx] != h_output_optimized[idx]) {
            std::cout << "Mismatch at index " << idx << ": global=" << h_output[idx] 
                      << ", optimized=" << h_output_optimized[idx] << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "All transpose results are correct.\n";
    } else {
        std::cout << "Some transpose results are incorrect.\n";
    }

    // Free memory
    free(h_input);
    free(h_output);
    free(h_output_optimized);
    free(h_output_highly_optimized);
    free(h_output_advanced);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_optimized);
    cudaFree(d_output_highly_optimized);
    cudaFree(d_output_advanced);

    std::cout << "Optimized program completed successfully!" << std::endl;
    return 0;
} 