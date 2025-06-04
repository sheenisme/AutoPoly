# AutoPoly: 基于MLIR的自动多面体调度框架

AutoPoly是一个构建在MLIR之上的综合性多面体调度框架，通过多面体模型变换提供自动循环优化。它采用三级分离架构，实现目标检测、调度策略选择和调度算法应用的自动化。

## 特性

### 核心功能
- **多面体模型提取**: 使用ISL将MLIR仿射方言操作转换为多面体模型
- **依赖分析**: 对数据、内存和控制依赖进行全面分析
- **三级分离架构**: 自动目标检测 → 策略选择 → 算法应用
- **多目标支持**: CPU、GPU、OpenCL、FPGA、CGRA、NPU、DPU、PIM
- **高级变换**: 分块、融合、倾斜、并行化、向量化

### 支持的变换
- **循环分块**: 基于目标内存层次结构的自动分块大小选择
- **循环融合**: 智能融合以改善数据局部性
- **循环并行化**: 基于依赖分析的自动并行循环生成
- **循环倾斜**: 依赖感知的倾斜变换
- **向量化**: 向量/SIMD优化提示
- **内存优化**: 数组私有化和内存合并

### 目标平台支持
- **CPU**: 具有缓存层次优化的多核处理器
- **GPU**: CUDA兼容的图形处理器
- **OpenCL**: OpenCL兼容设备
- **FPGA**: 现场可编程门阵列
- **CGRA**: 粗粒度可重构阵列
- **NPU**: 神经处理单元
- **DPU**: 深度处理单元
- **PIM**: 存内计算架构

## 架构设计

AutoPoly实现了三级分离架构：

```
┌─────────────────┐    ┌──────────────────┐     ┌─────────────────┐
│  目标           │    │  调度策略         │     │  调度算法        │
│  检测           │───>│  选择             │───> │                 │
│                 │    │                  │     │                 │
└─────────────────┘    └──────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐     ┌─────────────────┐
│ 硬件特征        │     │  目标特定优化     │     │ ISL, Feautrier, │
│ 和能力          │     │  参数            │     │ PLUTO, PPCG     │
└─────────────────┘    └──────────────────┘     └─────────────────┘
```

### 组件介绍

1. **目标检测模块** (`lib/Target/`)
   - 自动硬件检测和特征描述
   - 内存层次分析
   - 计算能力评估

2. **调度策略管理器** (`lib/Scheduling/`)
   - 目标特定优化策略选择
   - 基于硬件特征的参数调优
   - 算法选择逻辑

3. **多面体分析** (`lib/Analysis/`)
   - MLIR仿射到多面体模型转换
   - 全面的依赖分析
   - 内存访问模式分析

4. **调度变换** (`lib/Transform/`)
   - 多面体变换的实现
   - GPU优化的PPCG集成
   - 基于ISL的调度算法

5. **代码生成** (`lib/CodeGen/`)
   - 多面体调度到MLIR仿射转换
   - 并行循环生成
   - 内存访问优化

6. **MLIR遍历** (`lib/Passes/`)
   - 与MLIR遍历基础设施集成
   - 流水线管理
   - 遍历配置和编排

## 构建说明

### 前置条件
- LLVM/MLIR (版本 18+)
- ISL (整数集合库)
- PPCG (多面体并行代码生成器)
- CMake 3.20+
- C++17兼容编译器

### 构建步骤

1. **克隆仓库**:
   ```bash
   git clone https://github.com/sheenisme/AutoPoly.git
   cd AutoPoly
   git submodule update --init --recursive
   ```

2. **设置LLVM/MLIR构建目录**:
   ```bash
   export LLVM_BUILD_DIR=/path/to/your/llvm-build
   ```

3. **配置和构建**:
   ```bash
   mkdir build && cd build
   cmake -G Ninja .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_BUILD_DIR=${LLVM_BUILD_DIR}
   ninja
   ```

4. **运行测试**:
   ```bash
   ninja check-autopoly
   ```

## 使用

### 命令行工具

```bash
# 基本用法
./autopoly-mlir-opt input.mlir -autopoly -o output.mlir

# 指定目标
./autopoly-mlir-opt input.mlir -autopoly -target=gpu -o output.mlir

# 启用特定优化
./autopoly-mlir-opt input.mlir -autopoly -enable-tiling -enable-fusion -o output.mlir
```

### MLIR Pass 集成

```cpp
#include "AutoPoly/Transform/AutoPolyPass.h"

// 注册 pass
mlir::PassManager pm(context);
pm.addPass(autopoly::createAutoPolyPass());
```

### API 使用

```cpp
#include "AutoPoly/Scheduling/SchedulingFramework.h"

// 创建框架
autopoly::SchedulingFramework scheduler(context);

// 执行调度
auto newSchedule = scheduler.performScheduling(
    originalSchedule, dependences, autopoly::TargetType::GPU);
```

## 示例

### 矩阵乘法优化

输入:
```mlir
func.func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %0 = affine.load %A[%i, %k] : memref<1024x1024xf32>
        %1 = affine.load %B[%k, %j] : memref<1024x1024xf32>
        %2 = affine.load %C[%i, %j] : memref<1024x1024xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return
}
```

输出（CPU 目标）:
```mlir
func.func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.parallel (%i0) = (0) to (1024) step (32) {
    affine.parallel (%j0) = (0) to (1024) step (32) {
      affine.for %k0 = 0 to 1024 step 32 {
        affine.for %i = #map(%i0) to #map(%i0 + 32) {
          affine.for %j = #map(%j0) to #map(%j0 + 32) {
            affine.for %k = #map(%k0) to #map(%k0 + 32) {
              // 计算保持不变
            }
          }
        }
      }
    }
  }
  return
}
```

## 测试

### 单元测试
```bash
cd build
make check-autopoly-unittests
```

### 集成测试
```bash
cd build
make check-autopoly
```

### MLIR 测试
```bash
cd build
llvm-lit test/AutoPoly/
```

## 贡献

1. Fork 仓库
2. 创建功能分支
3. 为新功能添加测试
4. 确保所有测试通过
5. 提交 pull request

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 LICENSE 文件。

## 致谢

- 感谢 MLIR 和 LLVM 社区提供的基础设施和工具
- 感谢 PPCG 工具开发者提供的多面体编译优化技术
- 感谢 ISL 库开发者提供的数学优化库及其优化算法
- 感谢 PET 库开发者提供的多面体模型提取技术