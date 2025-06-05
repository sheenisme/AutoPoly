// RUN: autopoly-mlir-opt %s -autopoly-scheduling | FileCheck %s

func.func @codegen_basic(%A: memref<32xf32>, %B: memref<32xf32>) {
  affine.for %i = 0 to 32 {
    %a = affine.load %A[%i] : memref<32xf32>
    affine.store %a, %B[%i] : memref<32xf32>
  }
  return
}

// CHECK-LABEL: func.func @codegen_basic
// CHECK: affine.for
// CHECK: affine.load
// CHECK: affine.store 