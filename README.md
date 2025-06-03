[English](README.md) | [中文](README-zh.md)

---

<h1 align="center">autoStash</h1>

<p align="center">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status" />
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License" />
  <img src="https://img.shields.io/badge/stars-★--" alt="GitHub Stars" />
</p>

---

> **autoStash** is a C/C++ project for polyhedral-based automatic parallelization and code generation, integrating PPCG, PET, ISL, and LLVM/Clang/MLIR.

---

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Third-party Libraries](#third-party-libraries)
- [Getting Started](#getting-started)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [References & Credits](#references--credits)
- [License](#license)

---

## Features
- **Automatic extraction** of polyhedral regions from C code
- **Code generation** for CUDA and OpenCL
- **LLVM/Clang/MLIR integration** for modern C/C++
- **Out-of-tree build** for easy integration and maintenance

---

## Dependencies
| Name    | Version/Note                |
|---------|-----------------------------|
| CMake   | >= 3.20                     |
| GCC/Clang | C/C++17 support           |
| make    |                             |
| git     |                             |
| ninja   |                             |
| GMP     | GNU Multiple Precision      |
| LLVM/Clang/MLIR | Built from source   |
| MPFR, OpenMP, OpenCL | Optional       |

---

## Third-party Libraries
- [PPCG](https://repo.or.cz/ppcg.git): Polyhedral Parallel Code Generator
- [PET](https://repo.or.cz/pet.git): Polyhedral Extraction Tool
- [ISL](https://repo.or.cz/isl.git): Integer Set Library
- [LLVM Project](https://github.com/llvm/llvm-project): LLVM, Clang, MLIR

---

## Getting Started

> **Note:** Building LLVM/Clang/MLIR may take a long time and require significant disk space and memory.

### 1. Clone the repository
```sh
git clone --recursive <repo_url>
cd autoStash
```
If you forgot `--recursive`:
```sh
git submodule update --init --recursive
```

### 2. Build LLVM/Clang/MLIR
```sh
bash scripts/llvm-build.sh
```

### 3. Build autoStash
```sh
bash scripts/build.sh
```
- The main binary will be generated in `build/bin/autoStash`.

---

## Directory Structure
```text
├── CMakeLists.txt           # Main CMake configuration
├── cmake/                   # Custom CMake modules for dependencies
├── include/                 # Project headers
├── lib/                     # Project source files
├── scripts/                 # Build scripts
├── third_party/             # Submodules: ppcg, llvm-project, etc.
│   ├── ppcg/                # PPCG source and dependencies (ISL, PET)
│   └── llvm-project/        # LLVM, Clang, MLIR sources
├── build/                   # Build output (created after build)
```

---

## Usage
The main entry point is the `autoStash` binary. For example:
```sh
./build/bin/autoStash --help
```
You can use PPCG command-line options to process C files and generate CUDA/OpenCL code. See PPCG documentation for details.

---

## Troubleshooting
> **Common issues and solutions:**

| Problem | Solution |
|---------|----------|
| LLVM/Clang/MLIR build fails | Ensure enough disk space and memory. Use recent CMake and Ninja. |
| Missing dependencies | Install with: <br> `sudo apt install cmake ninja-build git build-essential libgmp-dev` |
| Submodule errors | `git submodule update --init --recursive` |
| Linker errors | Ensure all dependencies are built and paths are correct. |

---

## References & Credits
- [PPCG](https://repo.or.cz/ppcg.git)
- [PET](https://repo.or.cz/pet.git)
- [ISL](https://repo.or.cz/isl.git)
- [LLVM Project](https://github.com/llvm/llvm-project)

---

## License
This project is licensed under the Apache License 2.0 with LLVM Exceptions (SPDX: Apache-2.0 WITH LLVM-exception).

Third-party submodules (such as PPCG, PET, ISL, LLVM) are included for convenience and are licensed under their respective open source licenses. Please refer to each submodule's directory for details. 