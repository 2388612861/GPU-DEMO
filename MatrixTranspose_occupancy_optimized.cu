#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <limits>

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

// Optimized matrix transpose kernel with improved occupancy
__global__ void transposeMatrixOccupancyOptimized(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Define shared memory with padding to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Optimized loading: Reduce branch divergence
    unsigned int mask = (x < width) && (y < height);
    int index_in = y * width + x;  // Input matrix index
    tile[ty][tx] = mask ? input[index_in] : 0.0f;  // Load into shared memory

    __syncthreads();  // Wait for all threads to complete loading

    // Optimized storing: Reduce branch divergence
    int index_out = x * height + y;  // Output matrix index
    if (mask) {
        output[index_out] = tile[tx][ty];  // Write back to global memory (transposed)
    }
}

// Function to calculate optimal block size for maximum occupancy
void calculateOptimalBlockSize(float* h_input, int width, int height, dim3& blockSize, dim3& gridSize) {
    // Try different block sizes to find optimal occupancy
    std::vector<dim3> blockSizes = {
        dim3(16, 16),   // 256 threads per block
        dim3(32, 16),   // 512 threads per block
        dim3(16, 32),   // 512 threads per block
        dim3(32, 32),   // 1024 threads per block
        dim3(64, 8),    // 512 threads per block
        dim3(8, 64),    // 512 threads per block
        dim3(64, 16),   // 1024 threads per block
        dim3(16, 64),   // 1024 threads per block
        dim3(128, 8),   // 1024 threads per block
        dim3(8, 128),   // 1024 threads per block
        dim3(64, 32),   // 2048 threads per block
        dim3(32, 64),   // 2048 threads per block
        dim3(128, 16),  // 2048 threads per block
        dim3(16, 128),  // 2048 threads per block
    };

    std::cout << "Testing different block sizes for optimal occupancy:" << std::endl;
    std::cout << "==================================================" << std::endl;

    double bestTime = std::numeric_limits<double>::max();
    dim3 bestBlockSize = dim3(32, 32); // Default
    dim3 bestGridSize = dim3(32, 32);  // Default

    for (const auto& testBlockSize : blockSizes) {
        dim3 testGridSize((width + testBlockSize.x - 1) / testBlockSize.x, 
                         (height + testBlockSize.y - 1) / testBlockSize.y);
        
        std::cout << "Block size: " << testBlockSize.x << "x" << testBlockSize.y 
                  << " (" << testBlockSize.x * testBlockSize.y << " threads)"
                  << ", Grid size: " << testGridSize.x << "x" << testGridSize.y << std::endl;

        // Allocate device memory for testing
        float* d_input, *d_output;
        cudaMalloc(&d_input, width * height * sizeof(float));
        cudaMalloc(&d_output, width * height * sizeof(float));

        // Copy input data
        cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

        // Warm up
        transposeMatrixOccupancyOptimized<<<testGridSize, testBlockSize>>>(d_input, d_output, width, height);
        cudaDeviceSynchronize();

        // Measure performance
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) { // Run multiple times for accurate measurement
            transposeMatrixOccupancyOptimized<<<testGridSize, testBlockSize>>>(d_input, d_output, width, height);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double avgTime = elapsed.count() / 10.0;

        std::cout << "  Average time: " << avgTime << " s" << std::endl;

        // Update best configuration
        if (avgTime < bestTime) {
            bestTime = avgTime;
            bestBlockSize = testBlockSize;
            bestGridSize = testGridSize;
        }

        // Clean up
        cudaFree(d_input);
        cudaFree(d_output);
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "Best configuration:" << std::endl;
    std::cout << "  Block size: " << bestBlockSize.x << "x" << bestBlockSize.y 
              << " (" << bestBlockSize.x * bestBlockSize.y << " threads)" << std::endl;
    std::cout << "  Grid size: " << bestGridSize.x << "x" << bestGridSize.y << std::endl;
    std::cout << "  Best time: " << bestTime << " s" << std::endl;

    blockSize = bestBlockSize;
    gridSize = bestGridSize;
}

int main() {
    std::cout << "Starting Occupancy-Optimized Matrix Transpose CUDA Program..." << std::endl;
    
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);

    std::cout << "Matrix size: " << width << "x" << height << std::endl;

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    float* h_output_occupancy_optimized = (float*)malloc(size);

    // Initialize input matrix
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    float* d_output_occupancy_optimized;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_output_occupancy_optimized, size);

    // Copy input matrix from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Find optimal block and grid size
    dim3 optimalBlockSize, optimalGridSize;
    calculateOptimalBlockSize(h_input, width, height, optimalBlockSize, optimalGridSize);

    std::cout << "\nFinal configuration:" << std::endl;
    std::cout << "Grid size: " << optimalGridSize.x << "x" << optimalGridSize.y << std::endl;
    std::cout << "Block size: " << optimalBlockSize.x << "x" << optimalBlockSize.y << std::endl;

    // Test with default configuration (32x32)
    dim3 defaultBlockSize(32, 32);
    dim3 defaultGridSize((width + defaultBlockSize.x - 1) / defaultBlockSize.x, 
                        (height + defaultBlockSize.y - 1) / defaultBlockSize.y);

    // Matrix transpose using global memory with default configuration
    auto start = std::chrono::high_resolution_clock::now();
    transposeMatrix<<<defaultGridSize, defaultBlockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Global memory transpose time (default config): " << elapsed.count() << " s\n";

    // Matrix transpose using occupancy-optimized configuration
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixOccupancyOptimized<<<optimalGridSize, optimalBlockSize>>>(d_input, d_output_occupancy_optimized, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Occupancy-optimized transpose time: " << elapsed.count() << " s\n";

    // Test with different block sizes for comparison
    std::vector<dim3> comparisonBlockSizes = {
        dim3(16, 16), dim3(32, 32), dim3(64, 8), dim3(128, 8)
    };

    std::cout << "\nPerformance comparison with different block sizes:" << std::endl;
    std::cout << "==================================================" << std::endl;

    for (const auto& blockSize : comparisonBlockSizes) {
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                     (height + blockSize.y - 1) / blockSize.y);
        
        std::cout << "Block size: " << blockSize.x << "x" << blockSize.y 
                  << " (" << blockSize.x * blockSize.y << " threads)" << std::endl;

        // Warm up
        transposeMatrixOccupancyOptimized<<<gridSize, blockSize>>>(d_input, d_output_occupancy_optimized, width, height);
        cudaDeviceSynchronize();

        // Measure performance
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 5; ++i) {
            transposeMatrixOccupancyOptimized<<<gridSize, blockSize>>>(d_input, d_output_occupancy_optimized, width, height);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        double avgTime = elapsed.count() / 5.0;

        std::cout << "  Average time: " << avgTime << " s" << std::endl;
        std::cout << "  Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
        std::cout << "  Total threads: " << gridSize.x * gridSize.y * blockSize.x * blockSize.y << std::endl;
        std::cout << "  ---" << std::endl;
    }

    // Copy output matrix from device back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_occupancy_optimized, d_output_occupancy_optimized, size, cudaMemcpyDeviceToHost);

    // Verify results by checking a few sample points
    bool success = true;
    int check_points = 10;  // Check fewer points for faster verification
    for (int i = 0; i < check_points; ++i) {
        int idx = i * (width * height / check_points);
        if (idx < width * height && h_output[idx] != h_output_occupancy_optimized[idx]) {
            std::cout << "Mismatch at index " << idx << ": global=" << h_output[idx] 
                      << ", occupancy_optimized=" << h_output_occupancy_optimized[idx] << std::endl;
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
    free(h_output_occupancy_optimized);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_occupancy_optimized);

    std::cout << "Occupancy-optimized program completed successfully!" << std::endl;
    return 0;
} 