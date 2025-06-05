// RUN: autopoly-mlir-opt %s -polyhedral-analysis -dependence-analysis -verify-diagnostics | FileCheck %s

// Test complex affine expressions with parameters
func.func @parameterized_loops(%A: memref<?x?xf32>, %N: index, %M: index) {
  affine.for %i = 0 to %N {
    affine.for %j = 0 to %M {
      %val = affine.load %A[%i, %j] : memref<?x?xf32>
      %doubled = arith.mulf %val, arith.constant 2.0 : f32
      affine.store %doubled, %A[%i, %j] : memref<?x?xf32>
    }
  }
  return
}

// CHECK-LABEL: Polyhedral model extracted for function: parameterized_loops
// CHECK: Parametric loop bounds detected
// CHECK: Parameters: N, M
// CHECK: Domain: [N, M] -> { S[i, j] : 0 <= i < N and 0 <= j < M }

// Test strided access patterns
func.func @strided_access(%A: memref<1000xf32>, %B: memref<500xf32>) {
  affine.for %i = 0 to 500 {
    %val = affine.load %A[2 * %i] : memref<1000xf32>
    affine.store %val, %B[%i] : memref<500xf32>
  }
  return
}

// CHECK-LABEL: Polyhedral model extracted for function: strided_access
// CHECK: Strided access pattern detected
// CHECK: Stride: 2
// CHECK: Access map: { S[i] -> A[2*i] }

// Test multi-level affine expressions
func.func @complex_indexing(%A: memref<100x100x100xf32>) {
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
      affine.for %k = 0 to 100 {
        %idx1 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%i, %j, %k)
        %idx2 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1)>(%i, %j, %k)
        %idx3 = affine.apply affine_map<(d0, d1, d2) -> (d2)>(%i, %j, %k)
        
        // Ensure indices are within bounds
        %cond1 = arith.cmpi slt, %idx1, arith.constant 100 : index
        %cond2 = arith.cmpi slt, %idx2, arith.constant 100 : index
        
        scf.if %cond1 {
          scf.if %cond2 {
            %val = affine.load %A[%idx1, %idx2, %idx3] : memref<100x100x100xf32>
            %doubled = arith.mulf %val, arith.constant 2.0 : f32
            affine.store %doubled, %A[%i, %j, %k] : memref<100x100x100xf32>
          }
        }
      }
    }
  }
  return
}

// CHECK-LABEL: Polyhedral model extracted for function: complex_indexing
// CHECK: Complex affine expressions detected
// CHECK: Conditional accesses detected
// CHECK: Non-rectangular iteration space

// Test nested reductions with multiple arrays
func.func @nested_reductions(%A: memref<100x100xf32>, %row_sums: memref<100xf32>, %total: memref<1xf32>) {
  // Initialize total
  affine.store arith.constant 0.0 : f32, %total[0] : memref<1xf32>
  
  // Compute row sums and total
  affine.for %i = 0 to 100 {
    affine.store arith.constant 0.0 : f32, %row_sums[%i] : memref<100xf32>
    
    affine.for %j = 0 to 100 {
      %val = affine.load %A[%i, %j] : memref<100x100xf32>
      
      // Update row sum
      %current_row_sum = affine.load %row_sums[%i] : memref<100xf32>
      %new_row_sum = arith.addf %current_row_sum, %val : f32
      affine.store %new_row_sum, %row_sums[%i] : memref<100xf32>
      
      // Update total
      %current_total = affine.load %total[0] : memref<1xf32>
      %new_total = arith.addf %current_total, %val : f32
      affine.store %new_total, %total[0] : memref<1xf32>
    }
  }
  return
}

// CHECK-LABEL: Polyhedral model extracted for function: nested_reductions
// CHECK: Multiple reduction patterns detected
// CHECK: Reduction variables: row_sums, total
// CHECK: Nested reduction structure

// Test function with return values and iteration arguments
func.func @iter_args_computation(%A: memref<1000xf32>, %init: f32) -> f32 {
  %result = affine.for %i = 0 to 1000 iter_args(%acc = %init) -> f32 {
    %val = affine.load %A[%i] : memref<1000xf32>
    %new_acc = arith.addf %acc, %val : f32
    affine.yield %new_acc : f32
  }
  return %result : f32
}

// CHECK-LABEL: Polyhedral model extracted for function: iter_args_computation
// CHECK: Function with return value
// CHECK: Iteration arguments detected
// CHECK: Yield operation: f32
// CHECK: Reduction pattern with iter_args

func.func @complex_affine(%A: memref<32x32xf32>) {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %v = affine.load %A[%i, %j] : memref<32x32xf32>
      affine.store %v, %A[%i, %j] : memref<32x32xf32>
    }
  }
  return
}

// CHECK: Polyhedral model extracted
