//===- CPUTarget.cpp - CPU Target Optimization ---------------*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements CPU-specific optimizations and characteristics
// for the AutoPoly polyhedral scheduling framework.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Target/TargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <thread>

#define DEBUG_TYPE "cpu-target"

namespace autopoly {
namespace target {

/// CPU-specific optimization utilities
class CPUOptimizer {
public:
  /// Calculate optimal tile sizes for CPU cache hierarchy
  static std::vector<int> calculateCacheFriendlyTileSizes(
      const TargetCharacteristics& cpu,
      int num_dimensions,
      int element_size = 4) {
    
    std::vector<int> tile_sizes;
    
    // Find L1 cache for innermost tiles
    size_t l1_size = 32 * 1024; // Default 32KB
    for (const auto& mem : cpu.memory_hierarchy) {
      if (mem.level == MemoryLevel::LOCAL) {
        l1_size = mem.size_bytes;
        break;
      }
    }
    
    // Calculate tile size to fit in L1 cache
    // Assume working set = 3 arrays (A, B, C) for typical computation
    size_t available_cache = l1_size / 3;
    
    // For 2D: tile_size^2 * element_size <= available_cache
    // For 3D: tile_size^3 * element_size <= available_cache
    int base_tile_size;
    if (num_dimensions == 1) {
      base_tile_size = static_cast<int>(available_cache / element_size);
    } else if (num_dimensions == 2) {
      base_tile_size = static_cast<int>(std::sqrt(available_cache / element_size));
    } else if (num_dimensions == 3) {
      base_tile_size = static_cast<int>(std::cbrt(available_cache / element_size));
    } else {
      base_tile_size = static_cast<int>(std::pow(available_cache / element_size, 1.0 / num_dimensions));
    }
    
    // Ensure tile size is reasonable
    base_tile_size = std::max(4, std::min(128, base_tile_size));
    
    // Round to power of 2 for better performance
    int rounded_size = 1;
    while (rounded_size < base_tile_size) {
      rounded_size *= 2;
    }
    if (rounded_size > base_tile_size * 1.5) {
      rounded_size /= 2;
    }
    
    for (int i = 0; i < num_dimensions; ++i) {
      tile_sizes.push_back(rounded_size);
    }
    
    LLVM_DEBUG(llvm::dbgs() << "CPU cache-friendly tile sizes: ");
    for (int size : tile_sizes) {
      LLVM_DEBUG(llvm::dbgs() << size << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
    
    return tile_sizes;
  }

  /// Calculate optimal unroll factors for CPU vectorization
  static std::vector<int> calculateVectorizationUnrollFactors(
      const TargetCharacteristics& cpu,
      int num_loops) {
    
    std::vector<int> unroll_factors;
    
    int base_factor = 1;
    if (cpu.supports_vectorization) {
      // Detect vector width (simplified)
      if (cpu.vendor.find("Intel") != std::string::npos ||
          cpu.vendor.find("AMD") != std::string::npos) {
        base_factor = 8; // AVX-256 for 32-bit floats
      } else if (cpu.vendor.find("ARM") != std::string::npos) {
        base_factor = 4; // NEON 128-bit
      } else {
        base_factor = 4; // Conservative default
      }
    } else {
      base_factor = 2; // Minimal unrolling for non-vector CPUs
    }
    
    // Adjust based on number of compute units (avoid over-unrolling)
    if (cpu.compute_units < 4) {
      base_factor = std::min(base_factor, 4);
    }
    
    for (int i = 0; i < num_loops; ++i) {
      // Reduce unroll factor for inner loops
      int factor = base_factor;
      if (i == num_loops - 1) { // Innermost loop
        factor = base_factor;
      } else if (i == num_loops - 2) { // Second innermost
        factor = std::max(1, base_factor / 2);
      } else { // Outer loops
        factor = 1;
      }
      
      unroll_factors.push_back(factor);
    }
    
    LLVM_DEBUG(llvm::dbgs() << "CPU vectorization unroll factors: ");
    for (int factor : unroll_factors) {
      LLVM_DEBUG(llvm::dbgs() << factor << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
    
    return unroll_factors;
  }

  /// Calculate parallelization strategy for CPU
  static std::vector<int> calculateParallelizationStrategy(
      const TargetCharacteristics& cpu,
      int num_dimensions) {
    
    std::vector<int> parallel_dims;
    
    // For CPU, typically parallelize outer loops only
    // to avoid thread overhead and ensure good cache locality
    
    int max_parallel_loops = std::min(2, num_dimensions); // At most 2 parallel levels
    int available_cores = cpu.compute_units;
    
    if (available_cores >= 4) {
      // Enough cores for nested parallelism
      if (num_dimensions >= 2) {
        parallel_dims.push_back(0); // Outermost loop
        if (available_cores >= 16 && num_dimensions >= 3) {
          parallel_dims.push_back(1); // Second loop for many-core systems
        }
      } else if (num_dimensions == 1) {
        parallel_dims.push_back(0);
      }
    } else {
      // Limited cores, only parallelize outermost
      if (num_dimensions >= 1) {
        parallel_dims.push_back(0);
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "CPU parallelization strategy - parallel dimensions: ");
    for (int dim : parallel_dims) {
      LLVM_DEBUG(llvm::dbgs() << dim << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
    
    return parallel_dims;
  }

  /// Estimate CPU performance for given parameters
  static double estimatePerformance(
      const TargetCharacteristics& cpu,
      const std::vector<int>& tile_sizes,
      const std::vector<int>& parallel_dims,
      int total_operations) {
    
    // Simple performance model for CPU
    double compute_performance = cpu.peak_compute_throughput;
    
    // Cache efficiency factor
    double cache_efficiency = 1.0;
    if (!tile_sizes.empty()) {
      // Estimate cache hit rate based on tile sizes
      size_t l1_size = 32 * 1024; // Default
      for (const auto& mem : cpu.memory_hierarchy) {
        if (mem.level == MemoryLevel::LOCAL) {
          l1_size = mem.size_bytes;
          break;
        }
      }
      
      size_t working_set = 1;
      for (int size : tile_sizes) {
        working_set *= size;
      }
      working_set *= 4 * 3; // 3 arrays, 4 bytes per element
      
      if (working_set <= l1_size) {
        cache_efficiency = 0.95; // Excellent cache locality
      } else if (working_set <= l1_size * 8) {
        cache_efficiency = 0.80; // Good cache locality
      } else {
        cache_efficiency = 0.60; // Poor cache locality
      }
    }
    
    // Parallelization efficiency
    double parallel_efficiency = 1.0;
    if (!parallel_dims.empty()) {
      int parallel_work = 1;
      for (int dim : parallel_dims) {
        parallel_work *= cpu.compute_units; // Simplified
      }
      
      // Amdahl's law approximation
      double parallel_fraction = 0.9; // Assume 90% parallelizable
      parallel_efficiency = 1.0 / ((1.0 - parallel_fraction) + 
                                   parallel_fraction / std::min(parallel_work, cpu.compute_units));
    }
    
    double estimated_gflops = compute_performance * cache_efficiency * parallel_efficiency;
    
    LLVM_DEBUG(llvm::dbgs() << "CPU performance estimation:\n");
    LLVM_DEBUG(llvm::dbgs() << "  Base throughput: " << compute_performance << " GFLOPS\n");
    LLVM_DEBUG(llvm::dbgs() << "  Cache efficiency: " << cache_efficiency << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Parallel efficiency: " << parallel_efficiency << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Estimated performance: " << estimated_gflops << " GFLOPS\n");
    
    return estimated_gflops;
  }
};

/// CPU-specific scheduling parameters generator
class CPUSchedulingParametersGenerator {
public:
  static std::map<std::string, int> generateParameters(
      const TargetCharacteristics& cpu,
      int num_dimensions) {
    
    std::map<std::string, int> params;
    
    // Cache parameters
    size_t l1_size = 32 * 1024;
    size_t l2_size = 256 * 1024;
    for (const auto& mem : cpu.memory_hierarchy) {
      if (mem.level == MemoryLevel::LOCAL) {
        l1_size = mem.size_bytes;
      } else if (mem.level == MemoryLevel::SHARED) {
        l2_size = mem.size_bytes;
      }
    }
    
    params["l1_cache_size"] = static_cast<int>(l1_size);
    params["l2_cache_size"] = static_cast<int>(l2_size);
    params["cache_line_size"] = 64; // Typical cache line size
    
    // Parallelization parameters
    params["num_threads"] = cpu.compute_units;
    params["max_parallel_depth"] = cpu.compute_units >= 8 ? 2 : 1;
    
    // Vectorization parameters
    if (cpu.supports_vectorization) {
      params["vector_width"] = 8; // AVX-256 for floats
      params["enable_vectorization"] = 1;
    } else {
      params["vector_width"] = 1;
      params["enable_vectorization"] = 0;
    }
    
    // Loop transformation parameters
    params["enable_loop_interchange"] = 1;
    params["enable_loop_fusion"] = 1;
    params["enable_array_privatization"] = cpu.compute_units > 1 ? 1 : 0;
    
    // Prefetching parameters
    params["prefetch_distance"] = 16;
    params["enable_prefetching"] = 1;
    
    // Memory access optimization
    params["memory_coalescing_threshold"] = 0.8 * 100; // As percentage
    
    return params;
  }
};

} // namespace target
} // namespace autopoly
