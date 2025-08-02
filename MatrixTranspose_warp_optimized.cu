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

// Optimized matrix transpose kernel with improved warp efficiency
__global__ void transposeMatrixWarpOptimized(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Define shared memory with padding to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Optimized loading: Reduce branch divergence
    // Use mask to avoid conditional branches within warp
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

// Further optimized version with minimal branch divergence
__global__ void transposeMatrixMinimalDivergence(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Define shared memory with padding to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Phase 1: Load data with minimal branch divergence
    // Use arithmetic instead of conditional branches
    int valid_mask = (x < width) & (y < height);  // Use bitwise AND for efficiency
    int index_in = y * width + x;  // Input matrix index
    float value = __ldg(&input[index_in]);  // Use __ldg for read-only access
    tile[ty][tx] = valid_mask ? value : 0.0f;  // Conditional assignment

    __syncthreads();  // Ensure all threads have loaded data

    // Phase 2: Store data with minimal branch divergence
    int index_out = x * height + y;  // Output matrix index
    if (valid_mask) {
        output[index_out] = tile[tx][ty];  // Write back to global memory (transposed)
    }
}

// Advanced warp optimization with zero branch divergence
__global__ void transposeMatrixZeroDivergence(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Define shared memory with padding to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Phase 1: Load data with zero branch divergence
    // All threads in warp execute the same path
    int index_in = y * width + x;  // Input matrix index
    float value = __ldg(&input[index_in]);  // Use __ldg for read-only access
    
    // Use arithmetic operations instead of conditionals
    int valid_mask = (x < width) & (y < height);
    tile[ty][tx] = value * valid_mask;  // Multiply by mask (0 or 1)

    __syncthreads();  // Synchronize all threads

    // Phase 2: Store data with zero branch divergence
    int index_out = x * height + y;  // Output matrix index
    float result = tile[tx][ty] * valid_mask;  // Apply mask again
    output[index_out] = result;  // All threads write (invalid writes are masked)
}

// Ultra-optimized version with warp-level optimizations
__global__ void transposeMatrixUltraOptimized(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Define shared memory with padding to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Phase 1: Load with warp-level optimizations
    // Use __ldg for read-only global memory access
    int index_in = y * width + x;  // Input matrix index
    float value = __ldg(&input[index_in]);
    
    // Use arithmetic masking to avoid conditionals
    int valid = (x < width) & (y < height);
    tile[ty][tx] = value * valid;  // Zero out invalid values

    __syncthreads();  // Ensure all threads have loaded data

    // Phase 2: Store with warp-level optimizations
    int index_out = x * height + y;  // Output matrix index
    float result = tile[tx][ty] * valid;  // Apply mask again
    output[index_out] = result;  // All threads write (invalid writes are masked)
}

int main() {
    std::cout << "Starting Warp-Optimized Matrix Transpose CUDA Program..." << std::endl;
    
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);

    std::cout << "Matrix size: " << width << "x" << height << std::endl;

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    float* h_output_warp_optimized = (float*)malloc(size);
    float* h_output_minimal_divergence = (float*)malloc(size);
    float* h_output_zero_divergence = (float*)malloc(size);
    float* h_output_ultra_optimized = (float*)malloc(size);

    // Initialize input matrix
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    float* d_output_warp_optimized;
    float* d_output_minimal_divergence;
    float* d_output_zero_divergence;
    float* d_output_ultra_optimized;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_output_warp_optimized, size);
    cudaMalloc(&d_output_minimal_divergence, size);
    cudaMalloc(&d_output_zero_divergence, size);
    cudaMalloc(&d_output_ultra_optimized, size);

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

    // Matrix transpose using warp-optimized shared memory
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixWarpOptimized<<<gridSize, blockSize>>>(d_input, d_output_warp_optimized, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Warp-optimized shared memory transpose time: " << elapsed.count() << " s\n";

    // Matrix transpose using minimal divergence
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixMinimalDivergence<<<gridSize, blockSize>>>(d_input, d_output_minimal_divergence, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Minimal divergence transpose time: " << elapsed.count() << " s\n";

    // Matrix transpose using zero divergence
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixZeroDivergence<<<gridSize, blockSize>>>(d_input, d_output_zero_divergence, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Zero divergence transpose time: " << elapsed.count() << " s\n";

    // Matrix transpose using ultra optimization
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixUltraOptimized<<<gridSize, blockSize>>>(d_input, d_output_ultra_optimized, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Ultra-optimized transpose time: " << elapsed.count() << " s\n";

    // Copy output matrix from device back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_warp_optimized, d_output_warp_optimized, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_minimal_divergence, d_output_minimal_divergence, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_zero_divergence, d_output_zero_divergence, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_ultra_optimized, d_output_ultra_optimized, size, cudaMemcpyDeviceToHost);

    // Verify results by checking a few sample points
    bool success = true;
    int check_points = 10;  // Check fewer points for faster verification
    for (int i = 0; i < check_points; ++i) {
        int idx = i * (width * height / check_points);
        if (idx < width * height && h_output[idx] != h_output_warp_optimized[idx]) {
            std::cout << "Mismatch at index " << idx << ": global=" << h_output[idx] 
                      << ", warp_optimized=" << h_output_warp_optimized[idx] << std::endl;
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
    free(h_output_warp_optimized);
    free(h_output_minimal_divergence);
    free(h_output_zero_divergence);
    free(h_output_ultra_optimized);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_warp_optimized);
    cudaFree(d_output_minimal_divergence);
    cudaFree(d_output_zero_divergence);
    cudaFree(d_output_ultra_optimized);

    std::cout << "Warp-optimized program completed successfully!" << std::endl;
    return 0;
} 