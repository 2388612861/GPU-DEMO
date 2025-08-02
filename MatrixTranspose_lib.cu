#include "MatrixTranspose_lib.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <limits>
#include <cstring>

// Global variables for error handling
static char g_last_error[256] = "";
static bool g_cuda_initialized = false;

// CUDA kernels
__global__ void transposeMatrixBasic(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index_in = y * width + x;
        int index_out = x * height + y;
        output[index_out] = input[index_in];
    }
}

__global__ void transposeMatrixShared(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    unsigned int mask = (x < width) && (y < height);
    int index_in = y * width + x;
    tile[ty][tx] = mask ? input[index_in] : 0.0f;

    __syncthreads();

    int index_out = x * height + y;
    if (mask) {
        output[index_out] = tile[tx][ty];
    }
}

__global__ void transposeMatrixOccupancyOptimized(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    unsigned int mask = (x < width) && (y < height);
    int index_in = y * width + x;
    tile[ty][tx] = mask ? input[index_in] : 0.0f;

    __syncthreads();

    int index_out = x * height + y;
    if (mask) {
        output[index_out] = tile[tx][ty];
    }
}

__global__ void transposeMatrixFullyOptimized(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int valid_mask = (x < width) & (y < height);
    int index_in = y * width + x;
    float value = __ldg(&input[index_in]);
    tile[ty][tx] = value * valid_mask;

    __syncthreads();

    int index_out = x * height + y;
    float result = tile[tx][ty] * valid_mask;
    output[index_out] = result;
}

// Helper function to set error message
static void set_error(const char* error_msg) {
    strncpy(g_last_error, error_msg, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
}

// Initialize CUDA context
MATRIX_TRANSPOSE_API int matrix_transpose_init() {
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        set_error("Failed to set CUDA device");
        return MATRIX_TRANSPOSE_ERROR_INIT_FAILED;
    }
    
    g_cuda_initialized = true;
    return MATRIX_TRANSPOSE_SUCCESS;
}

// Cleanup CUDA context
MATRIX_TRANSPOSE_API void matrix_transpose_cleanup() {
    if (g_cuda_initialized) {
        cudaDeviceReset();
        g_cuda_initialized = false;
    }
}

// Get last error message
MATRIX_TRANSPOSE_API const char* matrix_transpose_get_last_error() {
    return g_last_error;
}

// Basic matrix transpose
MATRIX_TRANSPOSE_API int matrix_transpose_basic(float* input, float* output, int width, int height) {
    if (!g_cuda_initialized) {
        set_error("CUDA not initialized. Call matrix_transpose_init() first.");
        return MATRIX_TRANSPOSE_ERROR_INIT_FAILED;
    }

    if (!input || !output || width <= 0 || height <= 0) {
        set_error("Invalid parameters");
        return MATRIX_TRANSPOSE_ERROR_INVALID_PARAMETERS;
    }

    int size = width * height * sizeof(float);
    float *d_input, *d_output;

    cudaError_t error = cudaMalloc(&d_input, size);
    if (error != cudaSuccess) {
        set_error("Failed to allocate device memory for input");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_ALLOCATION;
    }

    error = cudaMalloc(&d_output, size);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        set_error("Failed to allocate device memory for output");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_ALLOCATION;
    }

    error = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Failed to copy input data to device");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_COPY;
    }

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    transposeMatrixBasic<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Kernel execution failed");
        return MATRIX_TRANSPOSE_ERROR_KERNEL_EXECUTION;
    }

    error = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Failed to copy output data from device");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_COPY;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return MATRIX_TRANSPOSE_SUCCESS;
}

// Shared memory matrix transpose
MATRIX_TRANSPOSE_API int matrix_transpose_shared(float* input, float* output, int width, int height) {
    if (!g_cuda_initialized) {
        set_error("CUDA not initialized. Call matrix_transpose_init() first.");
        return MATRIX_TRANSPOSE_ERROR_INIT_FAILED;
    }

    if (!input || !output || width <= 0 || height <= 0) {
        set_error("Invalid parameters");
        return MATRIX_TRANSPOSE_ERROR_INVALID_PARAMETERS;
    }

    int size = width * height * sizeof(float);
    float *d_input, *d_output;

    cudaError_t error = cudaMalloc(&d_input, size);
    if (error != cudaSuccess) {
        set_error("Failed to allocate device memory for input");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_ALLOCATION;
    }

    error = cudaMalloc(&d_output, size);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        set_error("Failed to allocate device memory for output");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_ALLOCATION;
    }

    error = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Failed to copy input data to device");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_COPY;
    }

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    transposeMatrixShared<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Kernel execution failed");
        return MATRIX_TRANSPOSE_ERROR_KERNEL_EXECUTION;
    }

    error = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Failed to copy output data from device");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_COPY;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return MATRIX_TRANSPOSE_SUCCESS;
}

// Occupancy optimized matrix transpose
MATRIX_TRANSPOSE_API int matrix_transpose_occupancy_optimized(float* input, float* output, int width, int height) {
    if (!g_cuda_initialized) {
        set_error("CUDA not initialized. Call matrix_transpose_init() first.");
        return MATRIX_TRANSPOSE_ERROR_INIT_FAILED;
    }

    if (!input || !output || width <= 0 || height <= 0) {
        set_error("Invalid parameters");
        return MATRIX_TRANSPOSE_ERROR_INVALID_PARAMETERS;
    }

    int size = width * height * sizeof(float);
    float *d_input, *d_output;

    cudaError_t error = cudaMalloc(&d_input, size);
    if (error != cudaSuccess) {
        set_error("Failed to allocate device memory for input");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_ALLOCATION;
    }

    error = cudaMalloc(&d_output, size);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        set_error("Failed to allocate device memory for output");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_ALLOCATION;
    }

    error = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Failed to copy input data to device");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_COPY;
    }

    // Use optimized block size (64x8 for better occupancy)
    dim3 blockSize(64, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    transposeMatrixOccupancyOptimized<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Kernel execution failed");
        return MATRIX_TRANSPOSE_ERROR_KERNEL_EXECUTION;
    }

    error = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Failed to copy output data from device");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_COPY;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return MATRIX_TRANSPOSE_SUCCESS;
}

// Fully optimized matrix transpose
MATRIX_TRANSPOSE_API int matrix_transpose_fully_optimized(float* input, float* output, int width, int height) {
    if (!g_cuda_initialized) {
        set_error("CUDA not initialized. Call matrix_transpose_init() first.");
        return MATRIX_TRANSPOSE_ERROR_INIT_FAILED;
    }

    if (!input || !output || width <= 0 || height <= 0) {
        set_error("Invalid parameters");
        return MATRIX_TRANSPOSE_ERROR_INVALID_PARAMETERS;
    }

    int size = width * height * sizeof(float);
    float *d_input, *d_output;

    cudaError_t error = cudaMalloc(&d_input, size);
    if (error != cudaSuccess) {
        set_error("Failed to allocate device memory for input");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_ALLOCATION;
    }

    error = cudaMalloc(&d_output, size);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        set_error("Failed to allocate device memory for output");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_ALLOCATION;
    }

    error = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Failed to copy input data to device");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_COPY;
    }

    // Use fully optimized configuration
    dim3 blockSize(128, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    transposeMatrixFullyOptimized<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Kernel execution failed");
        return MATRIX_TRANSPOSE_ERROR_KERNEL_EXECUTION;
    }

    error = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Failed to copy output data from device");
        return MATRIX_TRANSPOSE_ERROR_MEMORY_COPY;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return MATRIX_TRANSPOSE_SUCCESS;
}

// Performance test function
MATRIX_TRANSPOSE_API double matrix_transpose_performance_test(float* input, float* output, int width, int height, int method) {
    if (!g_cuda_initialized) {
        set_error("CUDA not initialized. Call matrix_transpose_init() first.");
        return -1.0;
    }

    if (!input || !output || width <= 0 || height <= 0) {
        set_error("Invalid parameters");
        return -1.0;
    }

    int size = width * height * sizeof(float);
    float *d_input, *d_output;

    cudaError_t error = cudaMalloc(&d_input, size);
    if (error != cudaSuccess) {
        set_error("Failed to allocate device memory for input");
        return -1.0;
    }

    error = cudaMalloc(&d_output, size);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        set_error("Failed to allocate device memory for output");
        return -1.0;
    }

    error = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Failed to copy input data to device");
        return -1.0;
    }

    dim3 blockSize, gridSize;
    
    switch (method) {
        case MATRIX_TRANSPOSE_METHOD_BASIC:
            blockSize = dim3(32, 32);
            break;
        case MATRIX_TRANSPOSE_METHOD_SHARED:
            blockSize = dim3(32, 32);
            break;
        case MATRIX_TRANSPOSE_METHOD_OCCUPANCY:
            blockSize = dim3(64, 8);
            break;
        case MATRIX_TRANSPOSE_METHOD_FULLY_OPTIMIZED:
            blockSize = dim3(128, 8);
            break;
        default:
            cudaFree(d_input);
            cudaFree(d_output);
            set_error("Invalid method");
            return -1.0;
    }

    gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Warm up
    switch (method) {
        case MATRIX_TRANSPOSE_METHOD_BASIC:
            transposeMatrixBasic<<<gridSize, blockSize>>>(d_input, d_output, width, height);
            break;
        case MATRIX_TRANSPOSE_METHOD_SHARED:
            transposeMatrixShared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
            break;
        case MATRIX_TRANSPOSE_METHOD_OCCUPANCY:
            transposeMatrixOccupancyOptimized<<<gridSize, blockSize>>>(d_input, d_output, width, height);
            break;
        case MATRIX_TRANSPOSE_METHOD_FULLY_OPTIMIZED:
            transposeMatrixFullyOptimized<<<gridSize, blockSize>>>(d_input, d_output, width, height);
            break;
    }

    cudaDeviceSynchronize();

    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10; ++i) {
        switch (method) {
            case MATRIX_TRANSPOSE_METHOD_BASIC:
                transposeMatrixBasic<<<gridSize, blockSize>>>(d_input, d_output, width, height);
                break;
            case MATRIX_TRANSPOSE_METHOD_SHARED:
                transposeMatrixShared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
                break;
            case MATRIX_TRANSPOSE_METHOD_OCCUPANCY:
                transposeMatrixOccupancyOptimized<<<gridSize, blockSize>>>(d_input, d_output, width, height);
                break;
            case MATRIX_TRANSPOSE_METHOD_FULLY_OPTIMIZED:
                transposeMatrixFullyOptimized<<<gridSize, blockSize>>>(d_input, d_output, width, height);
                break;
        }
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avgTime = elapsed.count() / 10.0;

    error = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        set_error("Failed to copy output data from device");
        return -1.0;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return avgTime;
} 