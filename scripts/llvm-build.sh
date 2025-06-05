#!/usr/bin/env bash
set -e

# Allow user to specify LLVM build directory as argument
if [ -n "$1" ]; then
    LLVM_BUILD_DIR="$1"
fi

# Set default LLVM build directory if not set
if [ -z "$LLVM_BUILD_DIR" ]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
    LLVM_BUILD_DIR="$PROJECT_ROOT/llvm-build"
fi

# Check if LLVM is already built (bin/llvm-config exists)
if [ -d "$LLVM_BUILD_DIR" ] && [ -x "$LLVM_BUILD_DIR/bin/llvm-config" ]; then
    echo "LLVM build directory already exists at $LLVM_BUILD_DIR, skipping LLVM build."
    exit 0
fi

echo "LLVM build directory: $LLVM_BUILD_DIR"

# Check required dependencies
for dep in cmake ninja git; do
    if ! command -v $dep &>/dev/null; then
        echo "Error: $dep is not installed." >&2
        exit 1
    fi
done

# Ensure llvm-project submodule is initialized
if [ ! -d "third_party/llvm-project/.git" ]; then
    echo "Initializing llvm-project submodule..."
    git submodule update --init --recursive third_party/llvm-project
else
    echo "Updating llvm-project submodule..."
    git submodule update --recursive third_party/llvm-project
fi

# Compute absolute path to LLVM source
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
LLVM_SRC_DIR="$PROJECT_ROOT/third_party/llvm-project/llvm"

# Build LLVM/Clang/MLIR
mkdir -p "$LLVM_BUILD_DIR"
cd "$LLVM_BUILD_DIR"
cmake -G Ninja "$LLVM_SRC_DIR" \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF
ninja
