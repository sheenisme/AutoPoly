

<h1 align="center">AutoPoly: Automatic Polyhedral Scheduling Framework for MLIR 
<img src="https://img.shields.io/badge/MLIR-Polyhedral-blue?logo=llvm&logoColor=white" alt="MLIR" height="24"/></h1>

<p align="center">
  <img src="https://img.shields.io/github/license/sheenisme/AutoPoly?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/github/workflow/status/sheenisme/AutoPoly/CI?label=CI&logo=github" alt="CI"/>
  <img src="https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B" alt="C++17"/>
  <img src="https://img.shields.io/badge/LLVM-18%2B-blueviolet?logo=llvm" alt="LLVM"/>
  <img src="https://img.shields.io/badge/ISL-supported-success?logo=gnu" alt="ISL"/>
</p>

[🇬🇧 English](README.md) | [🇨🇳 中文](README-zh.md)

## 🚀 Project Significance

Polyhedral compilation is a cornerstone of modern optimizing compilers, enabling aggressive loop transformations for high-performance computing and AI. 

**AutoPoly** provides:

- <img src="https://img.icons8.com/ios-filled/20/000000/parse-from-clipboard.png"/> **Automatic polyhedral model extraction** from MLIR affine dialects (ISL-based)
- <img src="https://img.icons8.com/ios-filled/20/000000/graph.png"/> **Comprehensive dependence analysis** (RAW, WAR, WAW, control, reduction)
- <img src="https://img.icons8.com/ios-filled/20/000000/chip.png"/> **Target-aware scheduling** for CPUs, GPUs, FPGAs, NPUs, DPUs, PIM, CGRA, and more
- <img src="https://img.icons8.com/ios-filled/20/000000/merge-git.png"/> **Rich transformation suite**: tiling, fusion, parallelization, skewing, vectorization
- <img src="https://img.icons8.com/ios-filled/20/000000/flow-chart.png"/> **Separation of concerns**: Target detection → Strategy selection → Algorithm application
- <img src="https://img.icons8.com/ios-filled/20/000000/plus-math.png"/> **Extensible design**: Easy to add new algorithms, targets, and analyses

## 🏛️ Architecture Overview

AutoPoly implements a three-level separation architecture:

<summary>Textual Architecture Diagram</summary>

```
┌───────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ Target        │   │ Scheduling         │   │ Scheduling         │
│ Detection     │-->| Strategy Selection │-->| Algorithm          │
└───────────────┘   └────────────────────┘   └────────────────────┘
        │                    │                        │
        ▼                    ▼                        ▼
┌───────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ Hardware      │   │ Target-Specific    │   │ ISL, Feautrier,    │
│ Characteristics│  │ Optimization Params│   │ PLUTO, PPCG, ...   │
└───────────────┘   └────────────────────┘   └────────────────────┘
```

### 📁 Project Structure

<details>
<summary>Click to expand</summary>

```
AutoPoly/
├── include/AutoPoly/          # C++ headers (modularized)
│   ├── Analysis/              # Polyhedral extraction, dependence analysis
│   ├── CodeGen/               # MLIR code generation from schedules
│   ├── Passes/                # MLIR pass infrastructure
│   ├── Scheduling/            # Scheduling strategies and algorithms
│   ├── Target/                # Target detection and characterization
│   └── Transform/             # Polyhedral transformations
├── lib/                       # C++ implementations
│   ├── ppcg_wrapper/          # C code optimizer (PPCG integration)
│   ├── Analysis/              # Analysis implementations
│   ├── CodeGen/               # Code generation implementations
│   ├── Passes/                # Pass implementations
│   ├── Scheduling/            # Scheduling implementations
│   ├── Target/                # Target detection implementations
│   └── Transform/             # Transformation implementations
├── tools/                     # Command-line tools
│   ├── autopoly-mlir-opt.cpp  # Main MLIR optimizer
│   └── autopoly-c-opt.cpp     # C code optimizer
├── scripts/                   # Build and install scripts
├── test/                      # Test files
├── unittests/                 # Unit tests
├── third_party/               # External dependencies (LLVM, ISL, PPCG, PET)
├── README.md                  # English documentation
├── README-zh.md               # Chinese documentation
└── LICENSE                    # License file
```
</details>

---

## 🧩 Key Components

- <img src="https://img.icons8.com/ios-filled/20/000000/inspection.png"/> **Analysis Framework**: Polyhedral extraction, dependence analysis, memory access analysis
- <img src="https://img.icons8.com/ios-filled/20/000000/search--v1.png"/> **Target Detection**: Automatic hardware detection, capability description, memory hierarchy analysis
- <img src="https://img.icons8.com/ios-filled/20/000000/strategy-board.png"/> **Scheduling Strategies**: Target-specific strategy selection, parameter tuning, extensible algorithm registry
- <img src="https://img.icons8.com/ios-filled/20/000000/synchronize.png"/> **Transformations**: Tiling, fusion, parallelization, skewing, vectorization, memory optimization
- <img src="https://img.icons8.com/ios-filled/20/000000/code.png"/> **Code Generation**: MLIR affine dialect emission, parallel loop generation, memory access optimization
- <img src="https://img.icons8.com/ios-filled/20/000000/flow-chart.png"/> **Pass Infrastructure**: MLIR pass registration, pipeline management, configuration

---

## ⚡ Installation

AutoPoly provides scripts for streamlined installation. You can use the provided scripts or follow the manual steps below.

```bash
# 1. Clone repository and submodules
git clone https://github.com/sheenisme/AutoPoly.git
cd AutoPoly
git submodule update --init --recursive

# 2. Build LLVM/MLIR (if not already available)
# (Recommended: let build.sh handle this automatically)

# 3. Build AutoPoly (recommended, will reuse cached LLVM if available)
bash scripts/build.sh /path/to/your/llvm-build
# or simply
bash scripts/build.sh
# (If omitted, will use ./llvm-build by default)

# 4. Run tests
ninja -C build check-autopoly

# 5. (Optional) Install
bash scripts/install.sh
```

> **Tip:** The build system will automatically detect and reuse an existing LLVM build (bin/llvm-config) in the specified directory, avoiding unnecessary recompilation. This is also used in CI for caching. See [CI workflow](.github/workflows/ci.yml) for details.

---

## 🛠️ Usage

### Command-Line Tool
```bash
# Basic usage (auto target detection)
autopoly-opt input.mlir -autopoly-scheduling

# Specify target type
autopoly-opt input.mlir -autopoly-scheduling="target-type=gpu"

# Custom tiling parameters
autopoly-opt input.mlir -autopoly-scheduling="tile-sizes=32,32,32"

# Enable specific optimizations
autopoly-opt input.mlir -autopoly-scheduling="enable-tiling=true enable-fusion=true"

# Debug mode with detailed output
autopoly-opt input.mlir -autopoly-scheduling="debug-mode=true dump-schedules=true"
```

### MLIR Pass Integration
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

## 🧪 Example: Matrix Multiplication Optimization

<details>
<summary>Show Example</summary>

**Input MLIR**:
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

**Optimized Output**:
```mlir
func.func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.parallel (%ii) = (0) to (1024) step (32) {
    affine.parallel (%jj) = (0) to (1024) step (32) {
      affine.for %kk = 0 to 1024 step 32 {
        affine.parallel (%i) = (%ii) to (min(1024, %ii + 32)) {
          affine.parallel (%j) = (%jj) to (min(1024, %jj + 32)) {
            affine.for %k = %kk to min(1024, %kk + 32) {
              // Optimized computation
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

## 🧑‍💻 Development & Debugging

- <img src="https://img.icons8.com/ios-filled/20/000000/bug.png"/> **Debug flags**: `export LLVM_DEBUG=autopoly-passes,polyhedral-extraction,scheduling-transform`
- <img src="https://img.icons8.com/ios-filled/20/000000/console.png"/> **ISL debug**: `export ISL_DEBUG=1`
- <img src="https://img.icons8.com/ios-filled/20/000000/speed.png"/> **Performance profiling**: `perf record ./build/bin/autopoly-mlir-opt --autopoly-scheduling input.mlir`
- <img src="https://img.icons8.com/ios-filled/20/000000/memory-slot.png"/> **Memory profiling**: `valgrind --tool=massif ./build/bin/autopoly-mlir-opt input.mlir`
- <img src="https://img.icons8.com/ios-filled/20/000000/code-file.png"/> **Code style**: Classes (PascalCase), functions (camelCase), variables (snake_case), constants (UPPER_SNAKE_CASE)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md] for details on code style, testing, and review process.

---

## 📚 Citation

If you use AutoPoly in your research, please cite:

```bibtex
@inproceedings{autopoly2024,
  title={AutoPoly: Automatic Polyhedral Scheduling for MLIR},
  author={Your Name},
  booktitle={Proceedings of ...},
  year={2024}
}
```

---

## 📝 License

This project is licensed under the **Apache License 2.0 with LLVM Exceptions**. See [LICENSE](LICENSE) for details.

> **Third-party submodules** (such as PPCG, PET, ISL, LLVM) are included for convenience and are licensed under their respective open source licenses. Please refer to each submodule's directory for details.

---

## 🙏 Acknowledgments

- LLVM/MLIR community for the excellent infrastructure
- ISL developers for the polyhedral library
- PPCG and PET teams for polyhedral extraction and GPU optimization
- Research groups advancing polyhedral compilation
