[English](README.md) | [中文](README-zh.md)

---

<h1 align="center">AutoPoly</h1>

<p align="center">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status" />
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License" />
  <img src="https://img.shields.io/badge/stars-★--" alt="GitHub Stars" />
</p>

---

> **AutoPoly** 是一个基于多面体模型的 C/C++ 自动并行化与代码生成工具，集成了 PPCG、PET、ISL 及 LLVM/Clang/MLIR。

---

## 目录
- [主要特性](#主要特性)
- [依赖环境](#依赖环境)
- [第三方库](#第三方库)
- [快速开始](#快速开始)
- [目录结构说明](#目录结构说明)
- [使用方法](#使用方法)
- [常见问题](#常见问题)
- [参考与致谢](#参考与致谢)
- [许可证](#许可证)

---

## 主要特性
- **自动提取** C 代码中的多面体区域（scop）
- **支持 CUDA 和 OpenCL** 代码生成
- **集成 LLVM/Clang/MLIR**，支持现代 C/C++ 语法
- **完全 out-of-tree 构建**，便于维护和集成

---

## 依赖环境
| 名称        | 版本/说明                |
|-------------|--------------------------|
| CMake       | >= 3.20                  |
| GCC/Clang   | 支持 C/C++17             |
| make        |                          |
| git         |                          |
| ninja       |                          |
| GMP         | GNU 多精度运算库         |
| LLVM/Clang/MLIR | 源码自动构建         |
| MPFR、OpenMP、OpenCL | 可选           |

---

## 第三方库
- [PPCG](https://repo.or.cz/ppcg.git)：多面体并行代码生成器
- [PET](https://repo.or.cz/pet.git)：多面体提取工具
- [ISL](https://repo.or.cz/isl.git)：整数集合库
- [LLVM Project](https://github.com/llvm/llvm-project)：LLVM、Clang、MLIR

---

## 快速开始

> **注意：** LLVM/Clang/MLIR 的编译耗时较长，且需要较多磁盘空间和内存。

### 1. 克隆仓库
```sh
git clone --recursive <repo_url>
cd AutoPoly
```
如果忘记加 `--recursive`：
```sh
git submodule update --init --recursive
```

### 2. 编译 LLVM/Clang/MLIR
```sh
bash scripts/llvm-build.sh
```

### 3. 编译 AutoPoly 主工程
```sh
bash scripts/build.sh
```
- 主程序会生成在 `build/bin/AutoPoly`。

---

## 目录结构说明
```text
├── CMakeLists.txt           # CMake 主配置文件
├── cmake/                   # 依赖相关的 CMake 模块
├── include/                 # 项目头文件
├── lib/                     # 项目源代码
├── scripts/                 # 构建脚本
├── third_party/             # 子模块：ppcg、llvm-project 等
│   ├── ppcg/                # PPCG 及其依赖（ISL、PET）
│   └── llvm-project/        # LLVM、Clang、MLIR 源码
├── build/                   # 构建输出目录（编译后生成）
```

---

## 使用方法
主程序入口为 `AutoPoly`，例如：
```sh
./build/bin/AutoPoly --help
```
你可以使用 PPCG 的命令行参数处理 C 文件并生成 CUDA/OpenCL 代码，具体参数和用法请参考 PPCG 官方文档。

---

## 常见问题
> **常见问题与解决方法：**

| 问题 | 解决方法 |
|------|----------|
| LLVM/Clang/MLIR 构建失败 | 请确保磁盘空间和内存充足，CMake 和 Ninja 版本较新。 |
| 依赖缺失 | 使用如下命令安装：<br> `sudo apt install cmake ninja-build git build-essential libgmp-dev` |
| 子模块错误 | `git submodule update --init --recursive` |
| 链接错误 | 请确保所有依赖均已正确编译，路径设置无误。 |

---

## 参考与致谢
- [PPCG](https://repo.or.cz/ppcg.git)
- [PET](https://repo.or.cz/pet.git)
- [ISL](https://repo.or.cz/isl.git)
- [LLVM Project](https://github.com/llvm/llvm-project)

---

## 许可证
本项目采用 Apache License 2.0 with LLVM Exceptions（SPDX: Apache-2.0 WITH LLVM-exception）开源协议。

第三方子模块（如 PPCG、PET、ISL、LLVM）仅为方便集成而收录，均遵循各自的开源协议，详情请参见各子模块目录下的 LICENSE 文件。 