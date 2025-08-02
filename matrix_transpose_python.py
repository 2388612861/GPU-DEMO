import ctypes
import numpy as np
import os
from typing import Optional, Tuple, Union
import time

class MatrixTransposeError(Exception):
    """Custom exception for matrix transpose errors"""
    pass

class MatrixTranspose:
    """
    Python wrapper for CUDA Matrix Transpose Library
    
    This class provides a high-level interface to the CUDA matrix transpose
    functions implemented in the dynamic library.
    """
    
    def __init__(self, lib_path: str = None):
        """
        Initialize the matrix transpose library
        
        Args:
            lib_path: Path to the dynamic library file
        """
        try:
            # Find the DLL file
            if lib_path is None:
                # Use absolute path as default
                lib_path = "F:\\CDUAtest\\MatrixTranspose\\MatrixTranspose\\MatrixTranspose_lib.dll"
                
                # Fallback to relative paths if absolute path doesn't exist
                if not os.path.exists(lib_path):
                    possible_paths = [
                        "MatrixTranspose_lib.dll",
                        os.path.join(os.path.dirname(__file__), "MatrixTranspose_lib.dll"),
                        os.path.join(os.getcwd(), "MatrixTranspose_lib.dll"),
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            lib_path = path
                            break
                    else:
                        raise MatrixTransposeError("Could not find MatrixTranspose_lib.dll in any of the expected locations")
            
            print(f"Loading library from: {lib_path}")
            
            # Load the dynamic library
            self.lib = ctypes.CDLL(lib_path)
            
            # Set function signatures
            self._setup_function_signatures()
            
            # Initialize CUDA context
            result = self.lib.matrix_transpose_init()
            if result != 0:  # MATRIX_TRANSPOSE_SUCCESS
                error_msg = self.lib.matrix_transpose_get_last_error()
                raise MatrixTransposeError(f"Failed to initialize CUDA: {error_msg.decode('utf-8')}")
                
            self._initialized = True
            print("Matrix Transpose library initialized successfully")
            
        except Exception as e:
            raise MatrixTransposeError(f"Failed to load library: {str(e)}")
    
    def _setup_function_signatures(self):
        """Setup function signatures for the dynamic library functions"""
        
        # Define argument types and return types
        self.lib.matrix_transpose_init.argtypes = []
        self.lib.matrix_transpose_init.restype = ctypes.c_int
        
        self.lib.matrix_transpose_cleanup.argtypes = []
        self.lib.matrix_transpose_cleanup.restype = None
        
        self.lib.matrix_transpose_get_last_error.argtypes = []
        self.lib.matrix_transpose_get_last_error.restype = ctypes.c_char_p
        
        # Matrix transpose functions
        transpose_funcs = [
            self.lib.matrix_transpose_basic,
            self.lib.matrix_transpose_shared,
            self.lib.matrix_transpose_occupancy_optimized,
            self.lib.matrix_transpose_fully_optimized
        ]
        
        for func in transpose_funcs:
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # input
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int,                    # width
                ctypes.c_int                     # height
            ]
            func.restype = ctypes.c_int
        
        # Performance test function
        self.lib.matrix_transpose_performance_test.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,                    # width
            ctypes.c_int,                    # height
            ctypes.c_int                     # method
        ]
        self.lib.matrix_transpose_performance_test.restype = ctypes.c_double
    
    def _check_initialized(self):
        """Check if the library is initialized"""
        if not hasattr(self, '_initialized') or not self._initialized:
            raise MatrixTransposeError("Library not initialized")
    
    def _validate_input(self, matrix: np.ndarray, expected_dtype=np.float32) -> Tuple[np.ndarray, int, int]:
        """
        Validate and prepare input matrix
        
        Args:
            matrix: Input matrix as numpy array
            expected_dtype: Expected data type
            
        Returns:
            Tuple of (matrix, width, height)
        """
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix")
        
        if matrix.dtype != expected_dtype:
            matrix = matrix.astype(expected_dtype)
        
        height, width = matrix.shape
        return matrix, width, height
    
    def _get_error_message(self) -> str:
        """Get the last error message from the library"""
        error_msg = self.lib.matrix_transpose_get_last_error()
        return error_msg.decode('utf-8') if error_msg else "Unknown error"
    
    def transpose_basic(self, matrix: np.ndarray) -> np.ndarray:
        """
        Perform basic matrix transpose
        
        Args:
            matrix: Input matrix (2D numpy array)
            
        Returns:
            Transposed matrix
        """
        self._check_initialized()
        matrix, width, height = self._validate_input(matrix)
        
        # Create output array
        output = np.empty((width, height), dtype=np.float32)
        
        # Get pointers to data
        input_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call the library function
        result = self.lib.matrix_transpose_basic(input_ptr, output_ptr, width, height)
        
        if result != 0:  # MATRIX_TRANSPOSE_SUCCESS
            raise MatrixTransposeError(f"Basic transpose failed: {self._get_error_message()}")
        
        return output
    
    def transpose_shared(self, matrix: np.ndarray) -> np.ndarray:
        """
        Perform matrix transpose with shared memory optimization
        
        Args:
            matrix: Input matrix (2D numpy array)
            
        Returns:
            Transposed matrix
        """
        self._check_initialized()
        matrix, width, height = self._validate_input(matrix)
        
        output = np.empty((width, height), dtype=np.float32)
        input_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        result = self.lib.matrix_transpose_shared(input_ptr, output_ptr, width, height)
        
        if result != 0:
            raise MatrixTransposeError(f"Shared memory transpose failed: {self._get_error_message()}")
        
        return output
    
    def transpose_occupancy_optimized(self, matrix: np.ndarray) -> np.ndarray:
        """
        Perform matrix transpose with occupancy optimization
        
        Args:
            matrix: Input matrix (2D numpy array)
            
        Returns:
            Transposed matrix
        """
        self._check_initialized()
        matrix, width, height = self._validate_input(matrix)
        
        output = np.empty((width, height), dtype=np.float32)
        input_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        result = self.lib.matrix_transpose_occupancy_optimized(input_ptr, output_ptr, width, height)
        
        if result != 0:
            raise MatrixTransposeError(f"Occupancy optimized transpose failed: {self._get_error_message()}")
        
        return output
    
    def transpose_fully_optimized(self, matrix: np.ndarray) -> np.ndarray:
        """
        Perform matrix transpose with all optimizations
        
        Args:
            matrix: Input matrix (2D numpy array)
            
        Returns:
            Transposed matrix
        """
        self._check_initialized()
        matrix, width, height = self._validate_input(matrix)
        
        output = np.empty((width, height), dtype=np.float32)
        input_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        result = self.lib.matrix_transpose_fully_optimized(input_ptr, output_ptr, width, height)
        
        if result != 0:
            raise MatrixTransposeError(f"Fully optimized transpose failed: {self._get_error_message()}")
        
        return output
    
    def performance_test(self, matrix: np.ndarray, method: int = 3) -> float:
        """
        Test performance of matrix transpose methods
        
        Args:
            matrix: Input matrix (2D numpy array)
            method: Method to test (0=basic, 1=shared, 2=occupancy, 3=fully_optimized)
            
        Returns:
            Execution time in seconds
        """
        self._check_initialized()
        matrix, width, height = self._validate_input(matrix)
        
        output = np.empty((width, height), dtype=np.float32)
        input_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        time_result = self.lib.matrix_transpose_performance_test(input_ptr, output_ptr, width, height, method)
        
        if time_result < 0:
            raise MatrixTransposeError(f"Performance test failed: {self._get_error_message()}")
        
        return time_result
    
    def benchmark_all_methods(self, matrix: np.ndarray) -> dict:
        """
        Benchmark all transpose methods
        
        Args:
            matrix: Input matrix (2D numpy array)
            
        Returns:
            Dictionary with method names and execution times
        """
        method_names = ["Basic", "Shared Memory", "Occupancy Optimized", "Fully Optimized"]
        results = {}
        
        for i, name in enumerate(method_names):
            try:
                time_result = self.performance_test(matrix, i)
                results[name] = time_result
            except MatrixTransposeError as e:
                results[name] = f"Error: {str(e)}"
        
        return results
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.cleanup()
    
    def cleanup(self):
        """Cleanup CUDA context"""
        if hasattr(self, '_initialized') and self._initialized:
            self.lib.matrix_transpose_cleanup()
            self._initialized = False
            print("Matrix Transpose library cleaned up")


# Convenience functions
def create_test_matrix(rows: int, cols: int) -> np.ndarray:
    """Create a test matrix with sequential values"""
    return np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)


def print_matrix(matrix: np.ndarray, name: str = "Matrix"):
    """Print a matrix in a formatted way"""
    print(f"{name} ({matrix.shape[0]}x{matrix.shape[1]}):")
    print(matrix)
    print()


def main():
    """Example usage of the MatrixTranspose class"""
    print("Matrix Transpose Python Interface Demo")
    print("=====================================")
    
    try:
        # Initialize the library
        with MatrixTranspose() as mt:
            # Create a small test matrix
            print("Creating test matrix...")
            test_matrix = create_test_matrix(4, 4)
            print_matrix(test_matrix, "Original Matrix")
            
            # Test all transpose methods
            methods = [
                ("Basic", mt.transpose_basic),
                ("Shared Memory", mt.transpose_shared),
                ("Occupancy Optimized", mt.transpose_occupancy_optimized),
                ("Fully Optimized", mt.transpose_fully_optimized)
            ]
            
            for name, method in methods:
                print(f"Testing {name} transpose...")
                try:
                    result = method(test_matrix)
                    print_matrix(result, f"{name} Transpose Result")
                except MatrixTransposeError as e:
                    print(f"{name} transpose failed: {e}")
                print()
            
            # Performance benchmark with larger matrix
            print("Performance Benchmark with 1024x1024 matrix:")
            print("=" * 50)
            
            large_matrix = create_test_matrix(1024, 1024)
            benchmark_results = mt.benchmark_all_methods(large_matrix)
            
            for method_name, time_result in benchmark_results.items():
                if isinstance(time_result, float):
                    print(f"{method_name}: {time_result:.6f} seconds")
                else:
                    print(f"{method_name}: {time_result}")
            
    except MatrixTransposeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main() 