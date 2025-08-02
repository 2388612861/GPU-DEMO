#ifndef MATRIX_TRANSPOSE_LIB_H
#define MATRIX_TRANSPOSE_LIB_H

#ifdef _WIN32
    #ifdef MATRIX_TRANSPOSE_EXPORTS
        #define MATRIX_TRANSPOSE_API __declspec(dllexport)
    #else
        #define MATRIX_TRANSPOSE_API __declspec(dllimport)
    #endif
#else
    #define MATRIX_TRANSPOSE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Matrix transpose function with basic configuration
MATRIX_TRANSPOSE_API int matrix_transpose_basic(float* input, float* output, int width, int height);

// Matrix transpose function with shared memory optimization
MATRIX_TRANSPOSE_API int matrix_transpose_shared(float* input, float* output, int width, int height);

// Matrix transpose function with occupancy optimization
MATRIX_TRANSPOSE_API int matrix_transpose_occupancy_optimized(float* input, float* output, int width, int height);

// Matrix transpose function with all optimizations
MATRIX_TRANSPOSE_API int matrix_transpose_fully_optimized(float* input, float* output, int width, int height);

// Performance test function - returns execution time in seconds
MATRIX_TRANSPOSE_API double matrix_transpose_performance_test(float* input, float* output, int width, int height, int method);

// Get last error message
MATRIX_TRANSPOSE_API const char* matrix_transpose_get_last_error();

// Initialize CUDA context
MATRIX_TRANSPOSE_API int matrix_transpose_init();

// Cleanup CUDA context
MATRIX_TRANSPOSE_API void matrix_transpose_cleanup();

// Method constants
#define MATRIX_TRANSPOSE_METHOD_BASIC 0
#define MATRIX_TRANSPOSE_METHOD_SHARED 1
#define MATRIX_TRANSPOSE_METHOD_OCCUPANCY 2
#define MATRIX_TRANSPOSE_METHOD_FULLY_OPTIMIZED 3

// Error codes
#define MATRIX_TRANSPOSE_SUCCESS 0
#define MATRIX_TRANSPOSE_ERROR_INIT_FAILED -1
#define MATRIX_TRANSPOSE_ERROR_MEMORY_ALLOCATION -2
#define MATRIX_TRANSPOSE_ERROR_INVALID_PARAMETERS -3
#define MATRIX_TRANSPOSE_ERROR_KERNEL_EXECUTION -4
#define MATRIX_TRANSPOSE_ERROR_MEMORY_COPY -5

#ifdef __cplusplus
}
#endif

#endif // MATRIX_TRANSPOSE_LIB_H 