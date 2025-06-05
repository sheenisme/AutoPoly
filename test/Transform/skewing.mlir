// RUN: autopoly-mlir-opt %s -autopoly-scheduling="enable-skewing=true" | FileCheck %s

func.func @skewing_test(%A: memref<32x32xf32>) {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %a = affine.load %A[%i, %j] : memref<32x32xf32>
      affine.store %a, %A[%i, %j] : memref<32x32xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @skewing_test
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.apply 