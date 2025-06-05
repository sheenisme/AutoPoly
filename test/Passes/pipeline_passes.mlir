// RUN: autopoly-mlir-opt %s -autopoly-polyhedral-analysis -autopoly-dependence-analysis -autopoly-scheduling | FileCheck %s

func.func @pipeline_test(%A: memref<16xf32>) {
  affine.for %i = 0 to 16 {
    %a = affine.load %A[%i] : memref<16xf32>
    affine.store %a, %A[%i] : memref<16xf32>
  }
  return
}

// CHECK-LABEL: func.func @pipeline_test
// CHECK: Polyhedral model extracted
// CHECK: Dependence analysis
// CHECK: affine.for 