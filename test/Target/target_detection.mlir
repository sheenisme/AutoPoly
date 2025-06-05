// RUN: autopoly-mlir-opt %s -autopoly-target-detection | FileCheck %s

module {
  func.func @dummy() {
    return
  }
}

// CHECK: Detected target type
// CHECK: Target characteristics 