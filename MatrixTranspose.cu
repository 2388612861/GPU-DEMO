#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// 定义矩阵转置的 kernel 函数
__global__ void transposeMatrix(float* input, float* output, int width, int height) {
    // 每个线程负责一个元素
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index_in = y * width + x;  // 输入矩阵的索引
        int index_out = x * height + y;  // 输出矩阵的索引
        output[index_out] = input[index_in];
    }
}

// 定义矩阵转置的 kernel 函数（使用共享内存）
__global__ void transposeMatrixShared(float* input, float* output, int width, int height) {
    __shared__ float tile[32][32];  // 定义共享内存

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (x < width && y < height) {
        int index_in = y * width + x;  // 输入矩阵的索引
        tile[tx][ty] = input[index_in];  // 加载到共享内存
    }

    __syncthreads();  // 等待所有线程完成加载

    if (x < width && y < height) {
        int index_out = x * height + y;  // 输出矩阵的索引
        output[index_out] = tile[ty][tx];  // 写回到全局内存
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);

    // 分配主机内存
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    float* h_output_shared = (float*)malloc(size);

    // 初始化输入矩阵
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // 分配设备内存
    float* d_input;
    float* d_output;
    float* d_output_shared;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_output_shared, size);

    // 将输入矩阵从主机复制到设备
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // 定义网格和块大小
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 使用全局内存进行矩阵转置
    auto start = std::chrono::high_resolution_clock::now();
    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Global memory transpose time: " << elapsed.count() << " s\n";

    // 使用共享内存进行矩阵转置
    start = std::chrono::high_resolution_clock::now();
    transposeMatrixShared<<<gridSize, blockSize>>>(d_input, d_output_shared, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Shared memory transpose time: " << elapsed.count() << " s\n";

    // 将输出矩阵从设备复制回主机
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_shared, d_output_shared, size, cudaMemcpyDeviceToHost);

    // 验证结果
    bool success = true;
    for (int i = 0; i < width * height; ++i) {
        if (h_output[i] != h_output_shared[i]) {
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Transpose result is correct.\n";
    } else {
        std::cout << "Transpose result is incorrect.\n";
    }

    // 释放内存
    free(h_input);
    free(h_output);
    free(h_output_shared);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_shared);

    return 0;
}