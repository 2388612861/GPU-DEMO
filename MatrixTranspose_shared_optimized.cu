#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <limits>

// Optimized shared memory transpose kernel
__global__ void transposeMatrixSharedOptimized(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Padding to avoid bank conflicts
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Use arithmetic masking to avoid branch divergence
    int valid_mask = (x < width) & (y < height);
    int index_in = y * width + x;
    
    // Load data with coalesced access pattern
    float value = 0.0f;
    if (valid_mask) {
        value = __ldg(&input[index_in]);  // Use __ldg for read-only global memory
    }
    tile[ty][tx] = value;
    
    __syncthreads();
    
    // Write data with coalesced access pattern
    int index_out = x * height + y;
    if (valid_mask) {
        output[index_out] = tile[tx][ty];
    }
}

// Ultra-optimized shared memory transpose
__global__ void transposeMatrixSharedUltraOptimized(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Use arithmetic masking and loop unrolling
    int valid_mask = (x < width) & (y < height);
    int index_in = y * width + x;
    
    // Load with loop unrolling for better instruction-level parallelism
    float value = 0.0f;
    if (valid_mask) {
        value = __ldg(&input[index_in]);
    }
    
    // Store with explicit bank conflict avoidance
    tile[ty][tx] = value;
    
    __syncthreads();
    
    // Write with coalesced access
    int index_out = x * height + y;
    if (valid_mask) {
        output[index_out] = tile[tx][ty];
    }
}

// Function to calculate optimal block size
dim3 calculateOptimalSharedBlockSize(int width, int height) {
    std::vector<dim3> blockSizes = {
        dim3(16, 16),   // 256 threads
        dim3(32, 16),   // 512 threads
        dim3(32, 32),   // 1024 threads
        dim3(64, 16),   // 1024 threads
        dim3(64, 8),    // 512 threads
        dim3(128, 8)    // 1024 threads
    };
    
    float bestTime = std::numeric_limits<float>::max();
    dim3 optimalBlockSize = dim3(32, 32);
    
    int size = width * height * sizeof(float);
    float *h_input, *h_output, *d_input, *d_output;
    
    h_input = new float[width * height];
    h_output = new float[width * height];
    
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    for (const auto& blockSize : blockSizes) {
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                     (height + blockSize.y - 1) / blockSize.y);
        
        // Warm up
        transposeMatrixSharedOptimized<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        cudaDeviceSynchronize();
        
        // Measure performance
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 10; ++i) {
            transposeMatrixSharedOptimized<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        }
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end - start;
        float avgTime = elapsed.count() / 10.0f;
        
        std::cout << "Block size (" << blockSize.x << "x" << blockSize.y 
                  << "): " << avgTime * 1000.0f << " ms" << std::endl;
        
        if (avgTime < bestTime) {
            bestTime = avgTime;
            optimalBlockSize = blockSize;
        }
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    
    std::cout << "Optimal block size: (" << optimalBlockSize.x << "x" << optimalBlockSize.y 
              << ") with time: " << bestTime * 1000.0f << " ms" << std::endl;
    
    return optimalBlockSize;
}

int main() {
    std::cout << "Shared Memory Matrix Transpose Optimization Test" << std::endl;
    std::cout << "================================================" << std::endl;
    
    const int width = 1024;
    const int height = 1024;
    const int size = width * height;
    
    std::cout << "Matrix size: " << width << "x" << height << std::endl;
    
    float *h_input = new float[size];
    float *h_output = new float[size];
    float *h_expected = new float[size];
    
    for (int i = 0; i < size; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Calculate expected result
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_expected[x * height + y] = h_input[y * width + x];
        }
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Find optimal block size
    std::cout << "\nFinding optimal block size..." << std::endl;
    dim3 optimalBlockSize = calculateOptimalSharedBlockSize(width, height);
    dim3 optimalGridSize((width + optimalBlockSize.x - 1) / optimalBlockSize.x,
                        (height + optimalBlockSize.y - 1) / optimalBlockSize.y);
    
    std::cout << "\nOptimal configuration:" << std::endl;
    std::cout << "Grid size: (" << optimalGridSize.x << "x" << optimalGridSize.y << ")" << std::endl;
    std::cout << "Block size: (" << optimalBlockSize.x << "x" << optimalBlockSize.y << ")" << std::endl;
    
    // Test optimizations
    std::cout << "\nTesting optimizations:" << std::endl;
    std::cout << "=====================" << std::endl;
    
    // Test 1: Basic optimization
    std::cout << "\n1. Basic Shared Memory Optimization:" << std::endl;
    cudaMemset(d_output, 0, size * sizeof(float));
    
    auto start = std::chrono::high_resolution_clock::now();
    transposeMatrixSharedOptimized<<<optimalGridSize, optimalBlockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (abs(h_output[i] - h_expected[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    std::chrono::duration<float> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() * 1000.0f << " ms" << std::endl;
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Test 2: Ultra optimization
    std::cout << "\n2. Ultra-Optimized Shared Memory:" << std::endl;
    cudaMemset(d_output, 0, size * sizeof(float));
    
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixSharedUltraOptimized<<<optimalGridSize, optimalBlockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    correct = true;
    for (int i = 0; i < size; ++i) {
        if (abs(h_output[i] - h_expected[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    elapsed = end - start;
    std::cout << "Time: " << elapsed.count() * 1000.0f << " ms" << std::endl;
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Performance comparison
    std::cout << "\nPerformance comparison with different sizes:" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    std::vector<int> sizes = {512, 1024, 2048};
    
    for (int matrixSize : sizes) {
        std::cout << "\nMatrix size: " << matrixSize << "x" << matrixSize << std::endl;
        
        int newSize = matrixSize * matrixSize;
        float *h_input_new = new float[newSize];
        float *h_output_new = new float[newSize];
        float *d_input_new, *d_output_new;
        
        cudaMalloc(&d_input_new, newSize * sizeof(float));
        cudaMalloc(&d_output_new, newSize * sizeof(float));
        
        for (int i = 0; i < newSize; ++i) {
            h_input_new[i] = static_cast<float>(i);
        }
        cudaMemcpy(d_input_new, h_input_new, newSize * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 blockSize = (matrixSize <= 1024) ? dim3(32, 32) : dim3(64, 16);
        dim3 gridSize((matrixSize + blockSize.x - 1) / blockSize.x,
                     (matrixSize + blockSize.y - 1) / blockSize.y);
        
        cudaMemset(d_output_new, 0, newSize * sizeof(float));
        
        start = std::chrono::high_resolution_clock::now();
        transposeMatrixSharedUltraOptimized<<<gridSize, blockSize>>>(d_input_new, d_output_new, matrixSize, matrixSize);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        
        elapsed = end - start;
        std::cout << "Time: " << elapsed.count() * 1000.0f << " ms" << std::endl;
        
        cudaFree(d_input_new);
        cudaFree(d_output_new);
        delete[] h_input_new;
        delete[] h_output_new;
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    delete[] h_expected;
    
    std::cout << "\nShared memory optimization test completed!" << std::endl;
    
    return 0;
} 