#!/usr/bin/env python3
"""
Simple example of using the Matrix Transpose Python interface

This script demonstrates how to use the CUDA matrix transpose library
from Python with minimal code.
"""

import numpy as np
from matrix_transpose_python import MatrixTranspose, MatrixTransposeError

def simple_example():
    """Simple example of matrix transpose"""
    print("Simple Matrix Transpose Example")
    print("===============================")
    
    # Create a test matrix
    matrix = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]], dtype=np.float32)
    
    print("Original matrix:")
    print(matrix)
    print()
    
    try:
        # Initialize the library
        with MatrixTranspose() as mt:
            # Perform transpose with different methods
            print("Transposing matrix...")
            
            # Basic transpose
            result_basic = mt.transpose_basic(matrix)
            print("Basic transpose result:")
            print(result_basic)
            print()
            
            # Fully optimized transpose
            result_optimized = mt.transpose_fully_optimized(matrix)
            print("Fully optimized transpose result:")
            print(result_optimized)
            print()
            
            # Verify results are the same
            if np.array_equal(result_basic, result_optimized):
                print("✓ Results match!")
            else:
                print("✗ Results don't match!")
                
    except MatrixTransposeError as e:
        print(f"Error: {e}")

def performance_example():
    """Performance comparison example"""
    print("\nPerformance Comparison Example")
    print("===============================")
    
    # Create a larger matrix for performance testing
    size = 1024
    matrix = np.random.rand(size, size).astype(np.float32)
    
    print(f"Testing with {size}x{size} matrix...")
    
    try:
        with MatrixTranspose() as mt:
            # Benchmark all methods
            results = mt.benchmark_all_methods(matrix)
            
            print("\nPerformance Results:")
            print("-" * 30)
            for method, time_result in results.items():
                if isinstance(time_result, float):
                    print(f"{method:20}: {time_result:.6f} seconds")
                else:
                    print(f"{method:20}: {time_result}")
                    
    except MatrixTransposeError as e:
        print(f"Error: {e}")

def numpy_comparison():
    """Compare with NumPy transpose"""
    print("\nNumPy Comparison Example")
    print("========================")
    
    # Create test matrix
    matrix = np.random.rand(512, 512).astype(np.float32)
    
    print("Comparing CUDA vs NumPy transpose...")
    
    try:
        with MatrixTranspose() as mt:
            # Time NumPy transpose
            import time
            start_time = time.time()
            numpy_result = matrix.T
            numpy_time = time.time() - start_time
            
            # Time CUDA transpose
            cuda_result = mt.transpose_fully_optimized(matrix)
            cuda_time = mt.performance_test(matrix, 3)  # Fully optimized method
            
            print(f"NumPy transpose time: {numpy_time:.6f} seconds")
            print(f"CUDA transpose time:  {cuda_time:.6f} seconds")
            print(f"Speedup: {numpy_time / cuda_time:.2f}x")
            
            # Verify results match
            if np.allclose(numpy_result, cuda_result):
                print("✓ Results match NumPy!")
            else:
                print("✗ Results don't match NumPy!")
                
    except MatrixTransposeError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run all examples
    simple_example()
    performance_example()
    numpy_comparison()
    
    print("\nAll examples completed!") 