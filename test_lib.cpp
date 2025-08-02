#include "MatrixTranspose_lib.h"
#include <iostream>
#include <iomanip>

void print_matrix(float* matrix, int width, int height, const char* name) {
    std::cout << name << ":" << std::endl;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::cout << std::setw(4) << matrix[y * width + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Matrix Transpose Library Test Program" << std::endl;
    std::cout << "=====================================" << std::endl;

    // Initialize CUDA
    int result = matrix_transpose_init();
    if (result != MATRIX_TRANSPOSE_SUCCESS) {
        std::cout << "Failed to initialize CUDA: " << matrix_transpose_get_last_error() << std::endl;
        return -1;
    }
    std::cout << "CUDA initialized successfully." << std::endl;

    // Test parameters
    const int width = 4;
    const int height = 4;
    const int size = width * height;

    // Allocate memory
    float* input = new float[size];
    float* output = new float[size];

    // Initialize input matrix
    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(i);
    }

    std::cout << "\nOriginal matrix:" << std::endl;
    print_matrix(input, width, height, "Input");

    // Test basic transpose
    std::cout << "Testing basic transpose..." << std::endl;
    result = matrix_transpose_basic(input, output, width, height);
    if (result == MATRIX_TRANSPOSE_SUCCESS) {
        print_matrix(output, height, width, "Basic Transpose Result");
    } else {
        std::cout << "Basic transpose failed: " << matrix_transpose_get_last_error() << std::endl;
    }

    // Test shared memory transpose
    std::cout << "Testing shared memory transpose..." << std::endl;
    result = matrix_transpose_shared(input, output, width, height);
    if (result == MATRIX_TRANSPOSE_SUCCESS) {
        print_matrix(output, height, width, "Shared Memory Transpose Result");
    } else {
        std::cout << "Shared memory transpose failed: " << matrix_transpose_get_last_error() << std::endl;
    }

    // Test occupancy optimized transpose
    std::cout << "Testing occupancy optimized transpose..." << std::endl;
    result = matrix_transpose_occupancy_optimized(input, output, width, height);
    if (result == MATRIX_TRANSPOSE_SUCCESS) {
        print_matrix(output, height, width, "Occupancy Optimized Transpose Result");
    } else {
        std::cout << "Occupancy optimized transpose failed: " << matrix_transpose_get_last_error() << std::endl;
    }

    // Test fully optimized transpose
    std::cout << "Testing fully optimized transpose..." << std::endl;
    result = matrix_transpose_fully_optimized(input, output, width, height);
    if (result == MATRIX_TRANSPOSE_SUCCESS) {
        print_matrix(output, height, width, "Fully Optimized Transpose Result");
    } else {
        std::cout << "Fully optimized transpose failed: " << matrix_transpose_get_last_error() << std::endl;
    }

    // Performance test with larger matrix
    std::cout << "\nPerformance Test with 1024x1024 matrix:" << std::endl;
    std::cout << "=========================================" << std::endl;

    const int large_width = 1024;
    const int large_height = 1024;
    const int large_size = large_width * large_height;

    float* large_input = new float[large_size];
    float* large_output = new float[large_size];

    // Initialize large matrix
    for (int i = 0; i < large_size; ++i) {
        large_input[i] = static_cast<float>(i);
    }

    // Test all methods
    const char* method_names[] = {
        "Basic",
        "Shared Memory",
        "Occupancy Optimized",
        "Fully Optimized"
    };

    for (int method = 0; method < 4; ++method) {
        double time = matrix_transpose_performance_test(large_input, large_output, large_width, large_height, method);
        if (time >= 0) {
            std::cout << method_names[method] << " method: " << std::fixed << std::setprecision(6) 
                      << time << " seconds" << std::endl;
        } else {
            std::cout << method_names[method] << " method failed: " << matrix_transpose_get_last_error() << std::endl;
        }
    }

    // Cleanup
    delete[] input;
    delete[] output;
    delete[] large_input;
    delete[] large_output;

    matrix_transpose_cleanup();
    std::cout << "\nTest completed successfully!" << std::endl;

    return 0;
} 