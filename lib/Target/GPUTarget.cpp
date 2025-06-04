//===- GPUTarget.cpp - GPU Target Optimization ---------------*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements GPU-specific optimizations and characteristics
// for the AutoPoly polyhedral scheduling framework.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Target/TargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

#define DEBUG_TYPE "gpu-target"

namespace autopoly {
namespace target {

/// GPU-specific optimization utilities
class GPUOptimizer {
public:
  /// Calculate thread-block friendly tile sizes for GPU
  static std::vector<int> calculateThreadBlockTileSizes(
      const TargetCharacteristics& gpu,
      int num_dimensions) {
    
    std::vector<int> tile_sizes;
    
    int max_threads = gpu.max_work_group_size;
    int warp_size = 32; // NVIDIA warp size (common)
    
    if (num_dimensions == 1) {
      // 1D: Use full thread block
      tile_sizes.push_back(std::min(max_threads, 1024));
    } else if (num_dimensions == 2) {
      // 2D: Create square blocks
      int tile_size = static_cast<int>(std::sqrt(max_threads));
      // Round down to nearest multiple of warp_size for X dimension
      int x_size = (tile_size / warp_size) * warp_size;
      if (x_size == 0) x_size = warp_size;
      
      int y_size = max_threads / x_size;
      
      tile_sizes = {x_size, y_size};
    } else if (num_dimensions == 3) {
      // 3D: Use smaller Z dimension
      int xy_threads = max_threads / 4; // Reserve Z=4
      int xy_size = static_cast<int>(std::sqrt(xy_threads));
      
      tile_sizes = {xy_size, xy_size, 4};
    } else {
      // General case: distribute threads
      int base_size = static_cast<int>(std::pow(max_threads, 1.0 / num_dimensions));
      for (int i = 0; i < num_dimensions; ++i) {
        tile_sizes.push_back(base_size);
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "GPU thread-block tile sizes: ");
    for (int size : tile_sizes) {
      LLVM_DEBUG(llvm::dbgs() << size << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
    
    return tile_sizes;
  }

  /// Calculate memory coalescing-friendly access patterns
  static std::vector<int> calculateCoalescingStrategy(
      const TargetCharacteristics& gpu,
      int num_dimensions) {
    
    std::vector<int> strategy;
    
    // For GPU, innermost dimension should have unit stride for coalescing
    // Prefer larger tiles in innermost dimension
    
    for (int i = 0; i < num_dimensions; ++i) {
      if (i == num_dimensions - 1) {
        // Innermost dimension: optimize for coalescing
        strategy.push_back(1); // Unit stride
      } else {
        // Outer dimensions: can have larger strides
        strategy.push_back(i + 1);
      }
    }
    
    return strategy;
  }

  /// Estimate GPU performance for given parameters
  static double estimateGPUPerformance(
      const TargetCharacteristics& gpu,
      const std::vector<int>& tile_sizes,
      int total_operations) {
    
    // GPU performance model
    double base_throughput = gpu.peak_compute_throughput;
    
    // Thread utilization efficiency
    double thread_efficiency = 1.0;
    if (!tile_sizes.empty()) {
      int total_threads = 1;
      for (int size : tile_sizes) {
        total_threads *= size;
      }
      
      // Check how well we utilize thread blocks
      int warp_size = 32;
      int warps_per_block = (total_threads + warp_size - 1) / warp_size;
      int utilized_threads = warps_per_block * warp_size;
      
      thread_efficiency = static_cast<double>(total_threads) / utilized_threads;
      
      // Penalize very small or very large blocks
      if (total_threads < 64) {
        thread_efficiency *= 0.7; // Too few threads
      } else if (total_threads > gpu.max_work_group_size) {
        thread_efficiency *= 0.5; // Exceeds limit
      }
    }
    
    // Memory coalescing efficiency
    double memory_efficiency = gpu.memory_coalescing_factor;
    
    // Occupancy estimation (simplified)
    double occupancy = std::min(1.0, static_cast<double>(gpu.compute_units) / 
                                     std::max(1, static_cast<int>(tile_sizes.size())));
    
    double estimated_tflops = base_throughput * thread_efficiency * 
                             memory_efficiency * occupancy / 1000.0; // Convert to TFLOPS
    
    LLVM_DEBUG(llvm::dbgs() << "GPU performance estimation:\n");
    LLVM_DEBUG(llvm::dbgs() << "  Base throughput: " << base_throughput << " GFLOPS\n");
    LLVM_DEBUG(llvm::dbgs() << "  Thread efficiency: " << thread_efficiency << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Memory efficiency: " << memory_efficiency << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Occupancy: " << occupancy << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Estimated performance: " << estimated_tflops << " TFLOPS\n");
    
    return estimated_tflops;
  }
};

/// GPU memory hierarchy optimizer
class GPUMemoryOptimizer {
public:
  /// Calculate shared memory usage strategy
  static std::map<std::string, int> calculateSharedMemoryStrategy(
      const TargetCharacteristics& gpu,
      const std::vector<int>& tile_sizes,
      int element_size = 4) {
    
    std::map<std::string, int> strategy;
    
    // Find shared memory size
    size_t shared_mem_size = 48 * 1024; // Default 48KB
    for (const auto& mem : gpu.memory_hierarchy) {
      if (mem.level == MemoryLevel::SHARED) {
        shared_mem_size = mem.size_bytes;
        break;
      }
    }
    
    // Calculate memory requirements for tiling
    size_t tile_memory = element_size;
    for (int size : tile_sizes) {
      tile_memory *= size;
    }
    
    // Estimate number of arrays that fit in shared memory
    int max_arrays = static_cast<int>(shared_mem_size / tile_memory);
    
    strategy["shared_memory_size"] = static_cast<int>(shared_mem_size);
    strategy["tile_memory_usage"] = static_cast<int>(tile_memory);
    strategy["max_shared_arrays"] = max_arrays;
    strategy["use_shared_memory"] = max_arrays >= 2 ? 1 : 0;
    
    // Global memory coalescing parameters
    strategy["coalescing_width"] = 128; // 128 bytes for optimal coalescing
    strategy["min_coalesced_access"] = 32; // Minimum threads for coalescing
    
    return strategy;
  }

  /// Calculate register usage strategy
  static std::map<std::string, int> calculateRegisterStrategy(
      const TargetCharacteristics& gpu) {
    
    std::map<std::string, int> strategy;
    
    // Estimate register file capacity (architecture dependent)
    int max_registers_per_thread = 255; // Common limit
    int target_occupancy = 75; // Target 75% occupancy
    
    strategy["max_registers_per_thread"] = max_registers_per_thread;
    strategy["target_occupancy"] = target_occupancy;
    strategy["enable_register_spilling"] = 0; // Avoid spilling
    
    return strategy;
  }
};

/// CUDA-specific optimizations
class CUDAOptimizer {
public:
  /// Generate CUDA-specific scheduling parameters
  static std::map<std::string, int> generateCUDAParameters(
      const TargetCharacteristics& gpu,
      int num_dimensions) {
    
    std::map<std::string, int> params;
    
    // CUDA block and grid dimensions
    params["max_threads_per_block"] = gpu.max_work_group_size;
    params["warp_size"] = 32;
    params["max_blocks_per_grid"] = 65535; // CUDA limit
    
    // Memory parameters
    auto shared_mem_strategy = GPUMemoryOptimizer::calculateSharedMemoryStrategy(gpu, {});
    for (const auto& param : shared_mem_strategy) {
      params[param.first] = param.second;
    }
    
    auto register_strategy = GPUMemoryOptimizer::calculateRegisterStrategy(gpu);
    for (const auto& param : register_strategy) {
      params[param.first] = param.second;
    }
    
    // Optimization flags
    params["enable_texture_memory"] = 0; // Conservative default
    params["enable_constant_memory"] = 1;
    params["prefer_shared_memory"] = 1;
    params["enable_coalescing_optimization"] = 1;
    
    // Compute capability dependent features
    params["supports_unified_memory"] = 1; // Modern GPUs
    params["supports_dynamic_parallelism"] = 0; // Conservative
    
    return params;
  }
};

/// OpenCL GPU-specific optimizations
class OpenCLGPUOptimizer {
public:
  /// Generate OpenCL GPU-specific parameters
  static std::map<std::string, int> generateOpenCLParameters(
      const TargetCharacteristics& gpu) {
    
    std::map<std::string, int> params;
    
    // OpenCL work group parameters
    params["max_work_group_size"] = gpu.max_work_group_size;
    params["max_work_item_dimensions"] = gpu.max_work_item_dimensions;
    
    for (size_t i = 0; i < gpu.max_work_item_sizes.size() && i < 3; ++i) {
      params["max_work_item_size_" + std::to_string(i)] = gpu.max_work_item_sizes[i];
    }
    
    // Memory parameters
    params["local_memory_size"] = gpu.supports_local_memory ? 32768 : 0;
    params["global_memory_size"] = static_cast<int>(8ULL * 1024 * 1024 * 1024); // 8GB default
    
    // Device capabilities
    params["supports_double_precision"] = gpu.supports_double_precision ? 1 : 0;
    params["supports_atomic_operations"] = gpu.supports_atomic_operations ? 1 : 0;
    
    // Optimization preferences
    params["prefer_local_memory"] = gpu.supports_local_memory ? 1 : 0;
    params["enable_vectorization"] = gpu.supports_vectorization ? 1 : 0;
    params["memory_coalescing_required"] = 1;
    
    return params;
  }
};

} // namespace target
} // namespace autopoly
