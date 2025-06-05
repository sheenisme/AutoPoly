
<h1 align="center">AutoPoly：面向MLIR的自动多面体调度框架 <img src="https://img.shields.io/badge/MLIR-Polyhedral-blue?logo=llvm&logoColor=white" alt="MLIR" height="24"/></h1>

<p align="center">
  <img src="https://img.shields.io/github/license/sheenisme/AutoPoly?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/github/workflow/status/sheenisme/AutoPoly/CI?label=CI&logo=github" alt="CI"/>
  <img src="https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B" alt="C++17"/>
  <img src="https://img.shields.io/badge/LLVM-18%2B-blueviolet?logo=llvm" alt="LLVM"/>
  <img src="https://img.shields.io/badge/ISL-supported-success?logo=gnu" alt="ISL"/>
</p>

[🇬🇧 English](README.md) | [🇨🇳 中文](README-zh.md)

## 🚀 项目意义

多面体编译是现代优化编译器的核心技术之一，能够实现高性能计算与AI场景下的高级循环变换。

**AutoPoly** 提供：

- <img src="https://img.icons8.com/ios-filled/20/000000/parse-from-clipboard.png"/> **自动多面体模型提取**（基于ISL）
- <img src="https://img.icons8.com/ios-filled/20/000000/graph.png"/> **全面依赖分析**（RAW、WAR、WAW、控制、归约）
- <img src="https://img.icons8.com/ios-filled/20/000000/chip.png"/> **面向目标的调度**（支持CPU、GPU、FPGA、NPU、DPU、PIM、CGRA等）
- <img src="https://img.icons8.com/ios-filled/20/000000/merge-git.png"/> **丰富的变换库**：分块、融合、并行化、倾斜、向量化
- <img src="https://img.icons8.com/ios-filled/20/000000/flow-chart.png"/> **三级分离设计**：硬件平台检测 → 策略选择 → 算法应用
- <img src="https://img.icons8.com/ios-filled/20/000000/plus-math.png"/> **易扩展设计**：便于集成新算法、目标和分析

---

## 🏛️ 架构概览

AutoPoly 实现了三级分离架构.

```
┌───────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ 硬件平台检测   │   │     策略选择        │   │     调度算法       │
└───────────────┘   └────────────────────┘   └────────────────────┘
        │                    │                        │
        ▼                    ▼                        ▼
┌───────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ 硬件特征与能力 │   │ 目标特定优化参数    │    │ ISL, Feautrier,    │
│               │   │                    │   │ PLUTO, PPCG 等     │
└───────────────┘   └────────────────────┘   └────────────────────┘
```

### 📁 项目结构

<details>
<summary>点击展开</summary>

```
AutoPoly/
├── include/AutoPoly/          # C++头文件（模块化）
│   ├── Analysis/              # 多面体提取与依赖分析
│   ├── CodeGen/               # 调度到MLIR代码生成
│   ├── Passes/                # MLIR Pass基础设施
│   ├── Scheduling/            # 调度策略与算法
│   ├── Target/                # 硬件平台检测与特征描述
│   └── Transform/             # 多面体变换
├── lib/                       # C++实现
│   ├── ppcg_wrapper/          # C代码优化器（PPCG集成）
│   ├── Analysis/              # 分析实现
│   ├── CodeGen/               # 代码生成实现
│   ├── Passes/                # Pass实现
│   ├── Scheduling/            # 调度实现
│   ├── Target/                # 硬件平台检测实现
│   └── Transform/             # 变换实现
├── tools/                     # 命令行工具
│   ├── autopoly-mlir-opt.cpp  # 主MLIR优化器
│   └── autopoly-c-opt.cpp     # C代码优化器
├── scripts/                   # 构建与安装脚本
├── test/                      # 测试文件
├── unittests/                 # 单元测试
├── third_party/               # 第三方依赖（LLVM, ISL, PPCG, PET）
├── README.md                  # 英文文档
├── README-zh.md               # 中文文档
└── LICENSE                    # 许可证
```
</details>

---

## 🧩 关键模块

- <img src="https://img.icons8.com/ios-filled/20/000000/inspection.png"/> **分析框架**：多面体提取、依赖分析、内存访问分析
- <img src="https://img.icons8.com/ios-filled/20/000000/search--v1.png"/> **硬件平台检测**：自动硬件检测、能力描述、内存层次分析
- <img src="https://img.icons8.com/ios-filled/20/000000/strategy-board.png"/> **调度策略**：目标特定策略选择、参数调优、算法注册
- <img src="https://img.icons8.com/ios-filled/20/000000/transform.png"/> **多面体变换**：分块、融合、并行化、倾斜、向量化、内存优化
- <img src="https://img.icons8.com/ios-filled/20/000000/code.png"/> **代码生成**：MLIR仿射方言生成、并行循环生成、内存访问优化
- <img src="https://img.icons8.com/ios-filled/20/000000/flow-chart.png"/> **Pass基础设施**：MLIR Pass注册、流水线管理、配置

---

## ⚡ 安装方法

AutoPoly 提供了便捷的脚本进行安装。可直接使用脚本或参考下述手动步骤。

```bash
# 1. 克隆仓库及子模块
git clone https://github.com/sheenisme/AutoPoly.git
cd AutoPoly
git submodule update --init --recursive

# 2. 构建LLVM/MLIR（如未安装）
bash scripts/llvm-build.sh
# 或手动设置LLVM_BUILD_DIR
export LLVM_BUILD_DIR=/path/to/your/llvm-build

# 3. 构建AutoPoly
bash scripts/build.sh

# 4. 运行测试
ninja -C build check-autopoly

# 5. （可选）安装
bash scripts/install.sh
```

> 详见 [CI workflow](.github/workflows/ci.yml) 了解自动化构建与测试流程。

---

## 🛠️ 使用方法

### 命令行工具
```bash
# 基本用法（自动硬件平台检测）
autopoly-opt input.mlir -autopoly-scheduling

# 指定目标类型
autopoly-opt input.mlir -autopoly-scheduling="target-type=gpu"

# 自定义分块参数
autopoly-opt input.mlir -autopoly-scheduling="tile-sizes=32,32,32"

# 启用特定优化
autopoly-opt input.mlir -autopoly-scheduling="enable-tiling=true enable-fusion=true"

# 调试模式与详细输出
autopoly-opt input.mlir -autopoly-scheduling="debug-mode=true dump-schedules=true"
```

### MLIR Pass集成
```cpp
#include "AutoPoly/Passes/AutoPolyPasses.h"
mlir::PassManager pm(&context);
autopoly::passes::AutoPolyPassOptions options;
options.target_type = "gpu";
options.enable_tiling = true;
options.tile_sizes = {32, 32, 32};
autopoly::passes::AutoPolyPipelineBuilder::addAutoPolyPasses(pm, options);
pm.run(module);
```

---

## 🧪 示例：矩阵乘法优化

<details>
<summary>点击展开示例</summary>

**输入MLIR**：
```mlir
func.func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
        %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
        %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return
}
```

**优化输出**：
```mlir
func.func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.parallel (%ii) = (0) to (1024) step (32) {
    affine.parallel (%jj) = (0) to (1024) step (32) {
      affine.for %kk = 0 to 1024 step 32 {
        affine.parallel (%i) = (%ii) to (min(1024, %ii + 32)) {
          affine.parallel (%j) = (%jj) to (min(1024, %jj + 32)) {
            affine.for %k = %kk to min(1024, %kk + 32) {
              // 优化后的计算
            }
          }
        }
      }
    }
  }
  return
}
```
</details>

---

## 🧑‍💻 开发与调试

- <img src="https://img.icons8.com/ios-filled/20/000000/bug.png"/> **调试标志**：`export LLVM_DEBUG=autopoly-passes,polyhedral-extraction,scheduling-transform`
- <img src="https://img.icons8.com/ios-filled/20/000000/console.png"/> **ISL调试**：`export ISL_DEBUG=1`
- <img src="https://img.icons8.com/ios-filled/20/000000/speed.png"/> **性能分析**：`perf record ./build/bin/autopoly-mlir-opt --autopoly-scheduling input.mlir`
- <img src="https://img.icons8.com/ios-filled/20/000000/memory-slot.png"/> **内存分析**：`valgrind --tool=massif ./build/bin/autopoly-mlir-opt input.mlir`
- <img src="https://img.icons8.com/ios-filled/20/000000/code-file.png"/> **代码风格**：类（PascalCase）、函数（camelCase）、变量（snake_case）、常量（UPPER_SNAKE_CASE）

---

## 🤝 贡献指南

欢迎贡献！详见 [CONTRIBUTING.md]，包括代码规范、测试与评审流程。

---

## 📚 学术引用

如在学术研究中使用AutoPoly，请引用：

```bibtex
@inproceedings{autopoly2024,
  title={AutoPoly: Automatic Polyhedral Scheduling for MLIR},
  author={Your Name},
  booktitle={Proceedings of ...},
  year={2024}
}
```

---

## 📝 许可证

本项目采用 **Apache License 2.0 with LLVM Exceptions** 许可，详见 [LICENSE](LICENSE)。

> **第三方子模块**（如PPCG、PET、ISL、LLVM）仅为方便集成，均遵循其各自开源协议，详情请参见各子模块目录。

---

## 🙏 致谢

- 感谢LLVM/MLIR社区提供的基础设施
- 感谢ISL开发者提供的多面体库
- 感谢PPCG与PET团队提供的多面体提取与GPU优化技术
- 感谢所有推动多面体编译技术进步的学术团体

