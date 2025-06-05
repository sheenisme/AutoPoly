// RUN: autopoly-mlir-opt %s -autopoly-scheduling="enable-tiling=true tile-sizes=32,32,32" | FileCheck %s

// Test basic matrix multiplication tiling
func.func @matrix_mult_tiling(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
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

// CHECK-LABEL: func.func @matrix_mult_tiling
// CHECK: affine.for %{{.*}} = 0 to 1024 step 32
// CHECK:   affine.for %{{.*}} = 0 to 1024 step 32
// CHECK:     affine.for %{{.*}} = 0 to 1024 step 32
// CHECK:       affine.for %{{.*}} = max(0, %{{.*}}) to min(1024, %{{.*}} + 32)
// CHECK:         affine.for %{{.*}} = max(0, %{{.*}}) to min(1024, %{{.*}} + 32)
// CHECK:           affine.for %{{.*}} = max(0, %{{.*}}) to min(1024, %{{.*}} + 32)

// Test tiling with automatic tile size selection
func.func @auto_tiling(%A: memref<512x512xf64>) {
  affine.for %i = 0 to 512 {
    affine.for %j = 0 to 512 {
      %val = affine.load %A[%i, %j] : memref<512x512xf64>
      %doubled = arith.mulf %val, arith.constant 2.0 : f64
      affine.store %doubled, %A[%i, %j] : memref<512x512xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @auto_tiling
// CHECK: affine.for %{{.*}} = 0 to 512 step {{[0-9]+}}
// CHECK:   affine.for %{{.*}} = 0 to 512 step {{[0-9]+}}
// CHECK:     affine.for %{{.*}} = max(0, %{{.*}}) to min(512, %{{.*}} + {{[0-9]+}})
// CHECK:       affine.for %{{.*}} = max(0, %{{.*}}) to min(512, %{{.*}} + {{[0-9]+}})

// Test tiling with non-square matrices
func.func @rectangular_tiling(%A: memref<2048x512xf32>, %B: memref<512xf32>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 512 {
      %a = affine.load %A[%i, %j] : memref<2048x512xf32>
      %b = affine.load %B[%j] : memref<512xf32>
      %prod = arith.mulf %a, %b : f32
      affine.store %prod, %A[%i, %j] : memref<2048x512xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @rectangular_tiling
// CHECK: affine.for %{{.*}} = 0 to 2048 step {{[0-9]+}}
// CHECK:   affine.for %{{.*}} = 0 to 512 step {{[0-9]+}}
// CHECK:     affine.for %{{.*}} = max(0, %{{.*}}) to min(2048, %{{.*}} + {{[0-9]+}})
// CHECK:       affine.for %{{.*}} = max(0, %{{.*}}) to min(512, %{{.*}} + {{[0-9]+}})

// RUN: autopoly-mlir-opt %s -autopoly-scheduling="enable-tiling=true tile-sizes=16" -o - | FileCheck %s

func.func @tiling_test(%A: memref<128xf32>) {
  affine.for %i = 0 to 128 {
    %v = affine.load %A[%i] : memref<128xf32>
    affine.store %v, %A[%i] : memref<128xf32>
  }
  return
}

// CHECK: affine.for
// CHECK: step 16
