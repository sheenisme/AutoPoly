// RUN: autopoly-mlir-opt %s -autopoly-scheduling="enable-fusion=true" | FileCheck %s

func.func @fusion_test(%A: memref<64xf32>, %B: memref<64xf32>, %C: memref<64xf32>) {
  affine.for %i = 0 to 64 {
    %a = affine.load %A[%i] : memref<64xf32>
    affine.store %a, %B[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %b = affine.load %B[%i] : memref<64xf32>
    affine.store %b, %C[%i] : memref<64xf32>
  }
  return
}

// CHECK-LABEL: func.func @fusion_test
// CHECK: affine.for
// CHECK: affine.for
// CHECK-NOT: affine.for 