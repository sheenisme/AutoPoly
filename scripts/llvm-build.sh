#!/usr/bin/env bash
set -e

# Check dependencies
for dep in cmake ninja git; do
    if ! command -v $dep &>/dev/null; then
        echo "Error: $dep is not installed." >&2
        exit 1
    fi
done

# Initialize and update submodules (if not already done)
if [ ! -d "third_party/llvm-project/.git" ]; then
    echo "Initializing llvm-project submodule..."
    git submodule update --init --recursive third_party/llvm-project
else
    echo "Updating llvm-project submodule..."
    git submodule update --recursive --remote third_party/llvm-project
fi

# Build LLVM/Clang/MLIR
mkdir -p llvm-build
cd llvm-build
cmake -G Ninja ../third_party/llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF
ninja
