# CUDA Matrix Transpose Optimization

高性能CUDA矩阵转置实现，包含多种优化策略和Python接口。

## 特性

- **多种优化策略**: 全局内存、共享内存、Warp效率、Occupancy优化
- **动态库支持**: 提供DLL接口供其他语言调用
- **Python接口**: 使用ctypes的Python包装器
- **性能测试**: 自动测试不同矩阵大小和线程块配置

## 文件结构

```
├── MatrixTranspose_shared_optimized.cu    # 最新优化版本
├── MatrixTranspose_occupancy_optimized.cu # Occupancy优化版本
├── MatrixTranspose_lib.h                  # 动态库头文件
├── MatrixTranspose_lib.cu                 # 动态库实现
├── test_lib.cpp                           # C++测试程序
├── matrix_transpose_python.py             # Python接口
├── example_usage.py                       # Python使用示例
└── requirements.txt                       # Python依赖
```

## 编译运行

### CUDA程序
```bash
nvcc -o MatrixTranspose_shared_optimized MatrixTranspose_shared_optimized.cu
./MatrixTranspose_shared_optimized
```

### 动态库
```bash
nvcc -shared -Xcompiler "/MD" -DMATRIX_TRANSPOSE_EXPORTS -o MatrixTranspose_lib.dll MatrixTranspose_lib.cu
```

### Python使用
```python
from matrix_transpose_python import MatrixTranspose

with MatrixTranspose() as mt:
    result = mt.transpose_shared(input_matrix)
```

## 优化技术

- **内存合并访问**: 优化全局内存读写模式
- **共享内存**: 使用tile缓存减少全局内存访问
- **Bank冲突避免**: 共享内存填充优化
- **Warp效率**: 减少分支发散
- **Occupancy优化**: 动态线程块大小选择

## 环境要求

- CUDA Toolkit 11.0+
- Python 3.7+ (可选)
- Visual Studio (Windows)
