// RUN: autopoly-mlir-opt %s -polyhedral-analysis -verify-diagnostics | FileCheck %s

// Test polyhedral extraction from simple affine loop nest
func.func @simple_matrix_mult(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
        %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
        %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return
}

// CHECK-LABEL: Polyhedral model extracted for function: simple_matrix_mult
// CHECK: Number of statements: 1
// CHECK: Number of arrays: 3
// CHECK: Loop nest depth: 3

// Test extraction from nested affine.if operations
func.func @conditional_computation(%A: memref<100x100xf32>, %B: memref<100x100xf32>) {
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
      affine.if affine_set<(d0, d1) : (d0 + d1 >= 0, d0 - d1 >= 0)>(%i, %j) {
        %a = affine.load %A[%i, %j] : memref<100x100xf32>
        %result = arith.mulf %a, %a : f32
        affine.store %result, %B[%i, %j] : memref<100x100xf32>
      }
    }
  }
  return
}

// CHECK-LABEL: Polyhedral model extracted for function: conditional_computation
// CHECK: Number of statements: 1
// CHECK: Conditional statement detected
// CHECK: Domain constraints: 2

// Test extraction from affine.parallel operations
func.func @parallel_computation(%A: memref<1000xf32>, %B: memref<1000xf32>) {
  affine.parallel (%i) = (0) to (1000) {
    %a = affine.load %A[%i] : memref<1000xf32>
    %result = arith.mulf %a, %a : f32
    affine.store %result, %B[%i] : memref<1000xf32>
  }
  return
}

// CHECK-LABEL: Polyhedral model extracted for function: parallel_computation
// CHECK: Number of statements: 1
// CHECK: Parallel dimension detected: 0

// Test extraction with yield operations
func.func @reduction_computation(%A: memref<1000xf32>) -> f32 {
  %sum = affine.for %i = 0 to 1000 iter_args(%iter = arith.constant 0.0 : f32) -> f32 {
    %a = affine.load %A[%i] : memref<1000xf32>
    %new_sum = arith.addf %iter, %a : f32
    affine.yield %new_sum : f32
  }
  return %sum : f32
}

// CHECK-LABEL: Polyhedral model extracted for function: reduction_computation
// CHECK: Number of statements: 1
// CHECK: Reduction operation detected
// CHECK: Return value: f32
