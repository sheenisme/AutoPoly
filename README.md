# AutoPoly: Automatic Polyhedral Scheduling Framework for MLIR

AutoPoly is a comprehensive polyhedral scheduling framework built on MLIR that provides automatic loop optimization through polyhedral model transformations. It features a three-tier separation architecture for target detection, scheduling strategy selection, and scheduling algorithm application.

## Features

### Core Capabilities
- **Polyhedral Model Extraction**: Converts MLIR affine dialect operations to polyhedral models using ISL
- **Dependence Analysis**: Comprehensive analysis of data, memory, and control dependences
- **Three-Tier Architecture**: Automatic target detection → strategy selection → algorithm application
- **Multi-Target Support**: CPU, GPU, OpenCL, FPGA, CGRA, NPU, DPU, PIM
- **Advanced Transformations**: Tiling, fusion, skewing, parallelization, vectorization

### Supported Transformations
- **Loop Tiling**: Automatic tile size selection based on target memory hierarchy
- **Loop Fusion**: Intelligent fusion to improve data locality
- **Loop Parallelization**: Automatic parallel loop generation with dependency analysis
- **Loop Skewing**: Dependency-aware skewing transformations
- **Vectorization**: Vector/SIMD optimization hints
- **Memory Optimization**: Array privatization and memory coalescing

### Target Platform Support
- **CPU**: Multi-core processors with cache hierarchy optimization
- **GPU**: CUDA-compatible graphics processors
- **OpenCL**: OpenCL-compatible devices
- **FPGA**: Field-Programmable Gate Arrays
- **CGRA**: Coarse-Grained Reconfigurable Arrays
- **NPU**: Neural Processing Units
- **DPU**: Deep Processing Units
- **PIM**: Processing-in-Memory architectures

## Architecture

AutoPoly implements a three-tier separation architecture:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Target         │     │  Scheduling      │     │  Scheduling     │
│  Detection      │ ──> │  Strategy        │ ──> │  Algorithms     │
│                 │     │  Selection       │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Hardware        │     │ Target-Specific  │     │ ISL, Feautrier, │
│ Characteristics │     │ Optimization     │     │ PLUTO, PPCG     │
│ & Capabilities  │     │ Parameters       │     │ Algorithms      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Components

1. **Target Detection Module** (`lib/Target/`)
   - Automatic hardware detection and characterization
   - Memory hierarchy analysis
   - Compute capability assessment

2. **Scheduling Strategy Manager** (`lib/Scheduling/`)
   - Target-specific optimization strategy selection
   - Parameter tuning based on hardware characteristics
   - Algorithm selection logic

3. **Polyhedral Analysis** (`lib/Analysis/`)
   - MLIR affine to polyhedral model conversion
   - Comprehensive dependence analysis
   - Memory access pattern analysis

4. **Scheduling Transformations** (`lib/Transform/`)
   - Implementation of polyhedral transformations
   - PPCG integration for GPU optimization
   - ISL-based scheduling algorithms

5. **Code Generation** (`lib/CodeGen/`)
   - Polyhedral schedule to MLIR affine conversion
   - Parallel loop generation
   - Memory access optimization

6. **MLIR Passes** (`lib/Passes/`)
   - Integration with MLIR pass infrastructure
   - Pipeline management
   - Pass configuration and orchestration

## Building

### Prerequisites
- LLVM/MLIR (version 18+)
- ISL (Integer Set Library)
- PPCG (Polyhedral Parallel Code Generator)
- CMake 3.20+
- C++17 compatible compiler

### Build Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sheenisme/AutoPoly.git
   cd AutoPoly
   git submodule update --init --recursive
   ```

2. **Set up LLVM/MLIR build directory**:
   ```bash
   export LLVM_BUILD_DIR=/path/to/your/llvm-build
   ```

3. **Configure and build**:
   ```bash
   mkdir build && cd build
   cmake -G Ninja .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_BUILD_DIR=${LLVM_BUILD_DIR}
   ninja
   ```

4. **Run tests**:
   ```bash
   ninja check-autopoly
   ```

## Usage

### Command Line Tool

The `autopoly-opt` tool provides MLIR optimization with polyhedral scheduling:

```bash
# Basic usage with auto target detection
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

### MLIR Pass Options

- `target-type`: Target hardware type (auto, cpu, gpu, opencl, fpga, cgra, npu, dpu, pim)
- `target-name`: Specific target device name
- `scheduling-algorithm`: Algorithm selection (auto, isl, feautrier, pluto, ppcg)
- `enable-tiling`: Enable loop tiling (default: true)
- `enable-fusion`: Enable loop fusion (default: true)
- `enable-parallelization`: Enable parallelization (default: true)
- `enable-skewing`: Enable loop skewing (default: false)
- `tile-sizes`: Explicit tile sizes (comma-separated)
- `max-parallel-depth`: Maximum parallel nesting depth (default: 3)

### Programming Interface

```cpp
#include "AutoPoly/Passes/AutoPolyPasses.h"

// Create pass manager with AutoPoly optimization
mlir::PassManager pm(&context);
autopoly::passes::AutoPolyPassOptions options;
options.target_type = "gpu";
options.enable_tiling = true;
options.tile_sizes = {32, 32, 32};

autopoly::passes::AutoPolyPipelineBuilder::addAutoPolyPasses(pm, options);
pm.run(module);
```

## Examples

### Matrix Multiplication Optimization

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

**Optimized Output** (with tiling and parallelization):
```mlir
func.func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.parallel (%ii) = (0) to (1024) step (32) {
    affine.parallel (%jj) = (0) to (1024) step (32) {
      affine.for %kk = 0 to 1024 step 32 {
        affine.parallel (%i) = (%ii) to (min(1024, %ii + 32)) {
          affine.parallel (%j) = (%jj) to (min(1024, %jj + 32)) {
            affine.for %k = %kk to min(1024, %kk + 32) {
              // Optimized computation with improved cache locality
            }
          }
        }
      }
    }
  }
  return
}
```

### Reduction Pattern Optimization

**Input**:
```mlir
func.func @reduction(%A: memref<1000xf32>) -> f32 {
  %sum = affine.for %i = 0 to 1000 iter_args(%acc = arith.constant 0.0 : f32) -> f32 {
    %val = affine.load %A[%i] : memref<1000xf32>
    %new_acc = arith.addf %acc, %val : f32
    affine.yield %new_acc : f32
  }
  return %sum : f32
}
```

**Optimized Output**:
```mlir
func.func @reduction(%A: memref<1000xf32>) -> f32 {
  %sum = affine.parallel (%i) = (0) to (1000) reduce ("addf") -> f32 {
    %val = affine.load %A[%i] : memref<1000xf32>
    affine.yield %val : f32
  }
  return %sum : f32
}
```

## Testing

The project includes comprehensive test suites:

### Analysis Tests
- Polyhedral model extraction from various affine constructs
- Dependence analysis validation
- Complex affine expression handling

### Transformation Tests
- Tiling correctness and performance
- Parallelization safety and efficiency
- Fusion opportunity detection
- Skewing transformation validation

### Integration Tests
- End-to-end optimization pipelines
- Target-specific optimization validation
- Performance regression testing

Run tests with:
```bash
# All tests
ninja check-autopoly

# Specific test categories
lit test/Analysis/
lit test/Transform/
lit test/Integration/
```

## Supported MLIR Constructs

AutoPoly supports optimization of the following MLIR affine dialect operations:

- `affine.for` - Standard affine loops with compile-time bounds
- `affine.parallel` - Parallel affine loops
- `affine.if` - Conditional execution with affine conditions
- `affine.yield` - Yield operations with iteration arguments
- `affine.load`/`affine.store` - Memory access operations
- Complex affine expressions and integer sets
- Nested loop structures with arbitrary depth
- Parametric loop bounds and access patterns

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and formatting
- Testing requirements
- Documentation standards
- Review process

### Development Setup

1. Follow the build instructions above
2. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
3. Run tests before submitting PRs:
   ```bash
   ninja check-autopoly
   ```

## Research and Publications

AutoPoly is based on state-of-the-art research in polyhedral compilation:

- Polyhedral model extraction and dependence analysis
- Target-aware scheduling strategy selection
- Integration of ISL, PPCG, and MLIR technologies

If you use AutoPoly in your research, please cite our work:

```bibtex
@inproceedings{autopoly2024,
  title={AutoPoly: Automatic Polyhedral Scheduling for MLIR},
  author={Your Name},
  booktitle={Proceedings of ...},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 with LLVM Exceptions. See [LICENSE](LICENSE) for details.

## Acknowledgments

- LLVM/MLIR community for the excellent infrastructure
- ISL developers for the polyhedral library
- PPCG team for GPU optimization techniques
- Research groups advancing polyhedral compilation

## Support

- **Issues**: [GitHub Issues](https://github.com/sheenisme/AutoPoly/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sheenisme/AutoPoly/discussions)
- **Documentation**: [Wiki](https://github.com/sheenisme/AutoPoly/wiki)

For questions about specific optimizations or target support, please include:
- MLIR input code
- Target hardware specification
- Expected vs. actual behavior
- AutoPoly version and build configuration
