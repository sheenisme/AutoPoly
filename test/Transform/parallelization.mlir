// RUN: autopoly-mlir-opt %s -autopoly-scheduling="enable-parallelization=true" | FileCheck %s

// Test basic parallel loop generation
func.func @parallel_computation(%A: memref<1000xf32>, %B: memref<1000xf32>) {
  affine.for %i = 0 to 1000 {
    %val = affine.load %A[%i] : memref<1000xf32>
    %squared = arith.mulf %val, %val : f32
    affine.store %squared, %B[%i] : memref<1000xf32>
  }
  return
}

// CHECK-LABEL: func.func @parallel_computation
// CHECK: affine.parallel (%{{.*}}) = (0) to (1000)
// CHECK:   %{{.*}} = affine.load %{{.*}}[%{{.*}}]
// CHECK:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}}
// CHECK:   affine.store %{{.*}}, %{{.*}}[%{{.*}}]

// Test matrix operations with parallel outer loops
func.func @parallel_matrix_ops(%A: memref<100x100xf32>, %B: memref<100x100xf32>, %C: memref<100x100xf32>) {
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
      %a = affine.load %A[%i, %j] : memref<100x100xf32>
      %b = affine.load %B[%i, %j] : memref<100x100xf32>
      %sum = arith.addf %a, %b : f32
      affine.store %sum, %C[%i, %j] : memref<100x100xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @parallel_matrix_ops
// CHECK: affine.parallel (%{{.*}}) = (0) to (100)
// CHECK:   affine.parallel (%{{.*}}) = (0) to (100)
// CHECK:     %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:     %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:     %{{.*}} = arith.addf %{{.*}}, %{{.*}}
// CHECK:     affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]

// Test that dependent loops are not parallelized
func.func @dependent_loops(%A: memref<100xf32>) {
  affine.for %i = 1 to 99 {
    %val = affine.load %A[%i - 1] : memref<100xf32>
    %doubled = arith.mulf %val, arith.constant 2.0 : f32
    affine.store %doubled, %A[%i] : memref<100xf32>
  }
  return
}

// CHECK-LABEL: func.func @dependent_loops
// CHECK: affine.for %{{.*}} = 1 to 99
// CHECK-NOT: affine.parallel

// Test reduction with parallel support
func.func @parallel_reduction(%A: memref<1000xf32>) -> f32 {
  %sum = affine.for %i = 0 to 1000 iter_args(%acc = arith.constant 0.0 : f32) -> f32 {
    %val = affine.load %A[%i] : memref<1000xf32>
    %new_acc = arith.addf %acc, %val : f32
    affine.yield %new_acc : f32
  }
  return %sum : f32
}

// CHECK-LABEL: func.func @parallel_reduction
// CHECK: %{{.*}} = affine.parallel (%{{.*}}) = (0) to (1000) reduce ("addf") -> f32
// CHECK:   %{{.*}} = affine.load %{{.*}}[%{{.*}}]
// CHECK:   affine.yield %{{.*}}

// Test nested parallelization with tiling
func.func @parallel_tiled_computation(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      %val = affine.load %A[%i, %j] : memref<1024x1024xf32>
      %sin_val = math.sin %val : f32
      affine.store %sin_val, %B[%i, %j] : memref<1024x1024xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @parallel_tiled_computation
// CHECK: affine.parallel (%{{.*}}) = (0) to (1024) step ({{[0-9]+}})
// CHECK:   affine.parallel (%{{.*}}) = (0) to (1024) step ({{[0-9]+}})
// CHECK:     affine.parallel (%{{.*}}) = (%{{.*}}) to (min(1024, %{{.*}} + {{[0-9]+}}))
// CHECK:       affine.parallel (%{{.*}}) = (%{{.*}}) to (min(1024, %{{.*}} + {{[0-9]+}}))
// CHECK:         %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:         %{{.*}} = math.sin %{{.*}}
// CHECK:         affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
