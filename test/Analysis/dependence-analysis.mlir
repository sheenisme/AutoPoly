// RUN: autopoly-mlir-opt %s -dependence-analysis -verify-diagnostics | FileCheck %s

// Test RAW (Read-After-Write) dependence analysis
func.func @raw_dependence(%A: memref<100xf32>) {
  affine.for %i = 1 to 99 {
    %val = affine.load %A[%i - 1] : memref<100xf32>
    %doubled = arith.mulf %val, arith.constant 2.0 : f32
    affine.store %doubled, %A[%i] : memref<100xf32>
  }
  return
}

// CHECK-LABEL: Dependence analysis for function: raw_dependence
// CHECK: RAW dependence detected
// CHECK: Distance vector: [1]
// CHECK: Carrying loop: 0
// CHECK: Blocks parallelization: true

// Test WAR (Write-After-Read) dependence analysis
func.func @war_dependence(%A: memref<100xf32>) {
  affine.for %i = 0 to 98 {
    %val = affine.load %A[%i + 1] : memref<100xf32>
    %doubled = arith.mulf %val, arith.constant 2.0 : f32
    affine.store %doubled, %A[%i] : memref<100xf32>
  }
  return
}

// CHECK-LABEL: Dependence analysis for function: war_dependence
// CHECK: WAR dependence detected
// CHECK: Distance vector: [-1]
// CHECK: Carrying loop: 0
// CHECK: Blocks parallelization: true

// Test WAW (Write-After-Write) dependence analysis
func.func @waw_dependence(%A: memref<100xf32>) {
  affine.for %i = 0 to 50 {
    affine.store arith.constant 1.0 : f32, %A[%i] : memref<100xf32>
    affine.store arith.constant 2.0 : f32, %A[%i] : memref<100xf32>
  }
  return
}

// CHECK-LABEL: Dependence analysis for function: waw_dependence
// CHECK: WAW dependence detected
// CHECK: Distance vector: [0]
// CHECK: Loop-independent dependence
// CHECK: Blocks parallelization: true

// Test complex multi-dimensional dependences
func.func @matrix_dependence(%A: memref<100x100xf32>) {
  affine.for %i = 1 to 99 {
    affine.for %j = 1 to 99 {
      %val1 = affine.load %A[%i - 1, %j] : memref<100x100xf32>
      %val2 = affine.load %A[%i, %j - 1] : memref<100x100xf32>
      %sum = arith.addf %val1, %val2 : f32
      affine.store %sum, %A[%i, %j] : memref<100x100xf32>
    }
  }
  return
}

// CHECK-LABEL: Dependence analysis for function: matrix_dependence
// CHECK: RAW dependence detected
// CHECK: Distance vector: [1, 0]
// CHECK: RAW dependence detected
// CHECK: Distance vector: [0, 1]
// CHECK: Carrying loops: 0, 1
// CHECK: Parallelizable dimensions: none

// Test independence analysis (no dependences)
func.func @independent_computation(%A: memref<100xf32>, %B: memref<100xf32>) {
  affine.for %i = 0 to 100 {
    %val = affine.load %A[%i] : memref<100xf32>
    %squared = arith.mulf %val, %val : f32
    affine.store %squared, %B[%i] : memref<100xf32>
  }
  return
}

// CHECK-LABEL: Dependence analysis for function: independent_computation
// CHECK: No dependences detected
// CHECK: Fully parallelizable
// CHECK: Parallelizable dimensions: [0]

// Test reduction pattern detection
func.func @reduction_pattern(%A: memref<1000xf32>, %sum: memref<1xf32>) {
  affine.for %i = 0 to 1000 {
    %val = affine.load %A[%i] : memref<1000xf32>
    %current_sum = affine.load %sum[0] : memref<1xf32>
    %new_sum = arith.addf %current_sum, %val : f32
    affine.store %new_sum, %sum[0] : memref<1xf32>
  }
  return
}

// CHECK-LABEL: Dependence analysis for function: reduction_pattern
// CHECK: Reduction dependence detected
// CHECK: Reduction variable: sum
// CHECK: Can be parallelized with reduction support
