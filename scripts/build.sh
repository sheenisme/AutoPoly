#!/usr/bin/env bash
set -e

# Check required dependencies
for dep in cmake ninja git; do
    if ! command -v $dep &>/dev/null; then
        echo "Error: $dep is not installed." >&2
        exit 1
    fi
done

# Initialize and update submodules
if [ ! -d "third_party/ppcg/.git" ] || [ ! -d "third_party/llvm-project/.git" ] || [ ! -d "third_party/googletest/.git" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
else
    echo "Updating git submodules..."
    git submodule update --recursive
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

# Allow user to specify LLVM build directory as argument
if [ -n "$1" ]; then
    LLVM_BUILD_DIR="$1"
fi

# Set default LLVM build directory if not set
if [ -z "$LLVM_BUILD_DIR" ]; then
    LLVM_BUILD_DIR="$PROJECT_ROOT/llvm-build"
fi

# Check if LLVM is already built (bin/llvm-config exists)
if [ -d "$LLVM_BUILD_DIR" ] && [ -x "$LLVM_BUILD_DIR/bin/llvm-config" ]; then
    echo "Using existing LLVM build at $LLVM_BUILD_DIR."
else
    echo "LLVM not found at $LLVM_BUILD_DIR, building LLVM..."
    export LLVM_BUILD_DIR
    bash scripts/llvm-build.sh "$LLVM_BUILD_DIR"
fi

echo "LLVM build directory: $LLVM_BUILD_DIR"

# Build main project with Ninja
mkdir -p build
cd build
cmake -G Ninja "$PROJECT_ROOT" -DLLVM_BUILD_DIR="$LLVM_BUILD_DIR"
ninja
