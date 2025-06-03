#!/usr/bin/env bash
set -e

# Check dependencies
for dep in cmake make git; do
    if ! command -v $dep &>/dev/null; then
        echo "Error: $dep is not installed." >&2
        exit 1
    fi
done

# Initialize and update submodules
if [ ! -d "third_party/ppcg/.git" ] || [ ! -d "third_party/llvm-project/.git" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
else
    echo "Updating git submodules..."
    git submodule update --recursive --remote
fi

# Build LLVM/Clang/MLIR
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

bash scripts/llvm-build.sh

# Build main project
mkdir -p build
cd build
cmake .. -DLLVM_BUILD_DIR="$PROJECT_ROOT/llvm-build"
make -j$(nproc)
