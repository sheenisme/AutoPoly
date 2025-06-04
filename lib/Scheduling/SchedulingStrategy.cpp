//===- SchedulingStrategy.cpp - Scheduling Strategy Framework -*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements the scheduling strategy framework that maps target
// hardware characteristics to appropriate scheduling algorithms and
// optimization parameters.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Scheduling/SchedulingStrategy.h"
#include "AutoPoly/Config.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <thread>

#define DEBUG_TYPE "scheduling-strategy"

namespace autopoly {
namespace scheduling {

// Implementation of CPUSchedulingStrategy
bool CPUSchedulingStrategy::isSuitableForTarget(const target::TargetCharacteristics& target) const {
  return target.type == target::TargetType::CPU;
}

SchedulingParameters CPUSchedulingStrategy::getParameters(
    const target::TargetCharacteristics& target) const {
  
  SchedulingParameters params;
  
  // CPU cache-friendly tiling
  params.enable_tiling = true;
  size_t l1_size = 32 * 1024; // Default 32KB L1 cache size
  if (params.tile_sizes.empty()) {
    // Calculate cache-friendly tile sizes
    for (const auto& mem : target.memory_hierarchy) {
      if (mem.level == target::MemoryLevel::LOCAL) {
        l1_size = mem.size_bytes;
        break;
      }
    }
    
    // Conservative tile size to fit in L1 cache
    int tile_size = static_cast<int>(std::sqrt(l1_size / (3 * 4))); // 3 arrays, 4 bytes per element
    tile_size = std::max(8, std::min(64, tile_size));
    
    params.tile_sizes = {tile_size, tile_size, tile_size};
  }
  
  // CPU parallelization strategy
  params.enable_nested_parallelism = target.compute_units >= 8;
  params.max_parallel_depth = target.compute_units >= 16 ? 3 : 2;
  
  // Conservative parallelization for CPU (outer loops only)
  params.parallel_dimensions = {0}; // Parallelize outermost loop
  if (target.compute_units >= 16) {
    params.parallel_dimensions.push_back(1); // Add second level for many-core
  }
  
  // Loop fusion for cache locality
  params.enable_loop_fusion = true;
  params.fusion_strategy = 1; // Maximal fusion for cache efficiency
  
  // Vectorization-aware unrolling
  params.enable_unrolling = target.supports_vectorization;
  if (target.supports_vectorization) {
    int vector_width = 8; // Assume AVX-256 for floats
    params.unroll_factors = {1, 1, vector_width}; // Unroll innermost loop
  } else {
    params.unroll_factors = {1, 1, 2}; // Minimal unrolling
  }
  
  // CPU doesn't typically need skewing
  params.enable_skewing = false;
  
  // Array privatization for multi-core
  params.enable_array_privatization = target.compute_units > 1;
  params.enable_copy_optimization = true;
  
  // Target-specific parameters
  params.target_specific_params["cache_line_size"] = 64;
  params.target_specific_params["l1_cache_size"] = static_cast<int>(l1_size);
  params.target_specific_params["num_cores"] = target.compute_units;
  params.target_specific_params["supports_vectorization"] = target.supports_vectorization ? 1 : 0;
  
  return params;
}

SchedulingAlgorithm CPUSchedulingStrategy::getPrimaryAlgorithm() const {
  return SchedulingAlgorithm::ISL_SCHEDULER; // ISL works well for CPU
}

std::vector<OptimizationTechnique> CPUSchedulingStrategy::getOptimizationTechniques() const {
  return {
    OptimizationTechnique::TILING,
    OptimizationTechnique::FUSION,
    OptimizationTechnique::PARALLELIZATION,
    OptimizationTechnique::VECTORIZATION,
    OptimizationTechnique::UNROLLING,
    OptimizationTechnique::PRIVATIZATION
  };
}

double CPUSchedulingStrategy::calculatePriority(const target::TargetCharacteristics& target) const {
  if (target.type != target::TargetType::CPU) return 0.0;
  
  // Priority based on CPU capabilities
  double priority = 5.0; // Base priority for CPU
  
  // Boost for more cores
  priority += std::log2(target.compute_units) * 0.5;
  
  // Boost for vectorization support
  if (target.supports_vectorization) priority += 1.0;
  
  // Boost for memory hierarchy
  priority += target.memory_hierarchy.size() * 0.2;
  
  return priority;
}

// Implementation of GPUSchedulingStrategy
bool GPUSchedulingStrategy::isSuitableForTarget(const target::TargetCharacteristics& target) const {
  return target.type == target::TargetType::GPU;
}

SchedulingParameters GPUSchedulingStrategy::getParameters(
    const target::TargetCharacteristics& target) const {
  
  SchedulingParameters params;
  
  // GPU thread-block friendly tiling
  params.enable_tiling = true;
  if (params.tile_sizes.empty()) {
    // Calculate thread-block friendly tile sizes
    int warp_size = 32; // NVIDIA warp size
    int max_threads = target.max_work_group_size;
    
    // Try to create square tiles that fit work group limits
    int tile_size = static_cast<int>(std::sqrt(max_threads));
    tile_size = std::min(tile_size, 32); // Don't exceed typical shared memory limits
    
    params.tile_sizes = {tile_size, tile_size, 1};
  }
  
  // Aggressive parallelization for GPU
  params.enable_nested_parallelism = true;
  params.max_parallel_depth = 3; // GPU can handle deep parallelism
  
  // Parallelize multiple dimensions
  params.parallel_dimensions = {0, 1}; // X and Y dimensions
  if (target.max_work_item_dimensions >= 3) {
    params.parallel_dimensions.push_back(2); // Z dimension if available
  }
  
  // Conservative fusion (may hurt parallelism)
  params.enable_loop_fusion = false;
  params.fusion_strategy = 2; // Minimal fusion
  
  // GPU doesn't typically benefit from unrolling (hardware handles it)
  params.enable_unrolling = false;
  params.unroll_factors = {1, 1, 1};
  
  // Skewing can help with memory coalescing
  params.enable_skewing = true;
  params.skewing_factors = {{1, 0}, {0, 1}}; // Identity for now
  
  // Array privatization less important for GPU
  params.enable_array_privatization = false;
  params.enable_copy_optimization = true; // Important for GPU memory hierarchy
  
  // GPU-specific parameters
  params.target_specific_params["warp_size"] = 32;
  params.target_specific_params["max_threads_per_block"] = target.max_work_group_size;
  params.target_specific_params["shared_memory_size"] = 48 * 1024; // 48KB typical
  params.target_specific_params["memory_coalescing_required"] = 1;
  
  return params;
}

SchedulingAlgorithm GPUSchedulingStrategy::getPrimaryAlgorithm() const {
  return SchedulingAlgorithm::PPCG_DEFAULT; // PPCG is designed for GPU
}

std::vector<OptimizationTechnique> GPUSchedulingStrategy::getOptimizationTechniques() const {
  return {
    OptimizationTechnique::TILING,
    OptimizationTechnique::PARALLELIZATION,
    OptimizationTechnique::SKEWING,
    OptimizationTechnique::VECTORIZATION
  };
}

double GPUSchedulingStrategy::calculatePriority(const target::TargetCharacteristics& target) const {
  if (target.type != target::TargetType::GPU) return 0.0;
  
  // High priority for GPU (often fastest)
  double priority = 10.0;
  
  // Boost for more compute units
  priority += std::log2(target.compute_units) * 0.1;
  
  // Boost for memory bandwidth
  for (const auto& mem : target.memory_hierarchy) {
    if (mem.level == target::MemoryLevel::GLOBAL) {
      priority += mem.bandwidth_gb_per_s * 0.001;
      break;
    }
  }
  
  return priority;
}

// Implementation of OpenCLSchedulingStrategy
bool OpenCLSchedulingStrategy::isSuitableForTarget(const target::TargetCharacteristics& target) const {
  return target.type == target::TargetType::OPENCL;
}

SchedulingParameters OpenCLSchedulingStrategy::getParameters(
    const target::TargetCharacteristics& target) const {
  
  SchedulingParameters params;
  
  // OpenCL work-group friendly tiling
  params.enable_tiling = true;
  if (params.tile_sizes.empty()) {
    int work_group_size = target.max_work_group_size;
    int tile_size = static_cast<int>(std::sqrt(work_group_size));
    tile_size = std::max(4, std::min(32, tile_size));
    
    params.tile_sizes = {tile_size, tile_size, 1};
  }
  
  // Moderate parallelization (OpenCL can be CPU or GPU)
  params.enable_nested_parallelism = target.compute_units >= 8;
  params.max_parallel_depth = 2;
  
  params.parallel_dimensions = {0, 1};
  
  // Conservative settings for portability
  params.enable_loop_fusion = true;
  params.fusion_strategy = 0; // Greedy fusion
  
  params.enable_unrolling = false;
  params.unroll_factors = {1, 1, 1};
  
  params.enable_skewing = false;
  
  params.enable_array_privatization = target.supports_local_memory;
  params.enable_copy_optimization = true;
  
  // OpenCL-specific parameters
  params.target_specific_params["work_group_size"] = target.max_work_group_size;
  params.target_specific_params["local_memory_size"] = target.supports_local_memory ? 16384 : 0;
  params.target_specific_params["global_memory_coalescing"] = 1;
  
  return params;
}

SchedulingAlgorithm OpenCLSchedulingStrategy::getPrimaryAlgorithm() const {
  return SchedulingAlgorithm::ISL_SCHEDULER; // ISL for portability
}

std::vector<OptimizationTechnique> OpenCLSchedulingStrategy::getOptimizationTechniques() const {
  return {
    OptimizationTechnique::TILING,
    OptimizationTechnique::PARALLELIZATION,
    OptimizationTechnique::FUSION
  };
}

double OpenCLSchedulingStrategy::calculatePriority(const target::TargetCharacteristics& target) const {
  if (target.type != target::TargetType::OPENCL) return 0.0;
  
  // Moderate priority (portable but not always optimal)
  double priority = 7.0;
  
  priority += std::log2(target.compute_units) * 0.3;
  
  if (target.supports_local_memory) priority += 1.0;
  
  return priority;
}

// Implementation of FPGASchedulingStrategy
bool FPGASchedulingStrategy::isSuitableForTarget(const target::TargetCharacteristics& target) const {
  return target.type == target::TargetType::FPGA;
}

SchedulingParameters FPGASchedulingStrategy::getParameters(
    const target::TargetCharacteristics& target) const {
  
  SchedulingParameters params;
  
  // FPGA pipeline-friendly tiling
  params.enable_tiling = true;
  if (params.tile_sizes.empty()) {
    // Smaller tiles for resource constraints
    params.tile_sizes = {16, 16, 16};
  }
  
  // Limited parallelization due to resource constraints
  params.enable_nested_parallelism = false;
  params.max_parallel_depth = 1;
  
  params.parallel_dimensions = {0}; // Single level parallelization
  
  // Aggressive fusion for pipeline efficiency
  params.enable_loop_fusion = true;
  params.fusion_strategy = 1; // Maximal fusion
  
  // High unrolling for pipeline efficiency
  params.enable_unrolling = true;
  params.unroll_factors = {1, 1, 8}; // Aggressive inner loop unrolling
  
  // Skewing can help with data flow
  params.enable_skewing = true;
  
  // Array privatization important for FPGA
  params.enable_array_privatization = true;
  params.enable_copy_optimization = true;
  
  // FPGA-specific parameters
  params.target_specific_params["pipeline_depth"] = 16;
  params.target_specific_params["resource_utilization_target"] = 80; // 80% utilization
  params.target_specific_params["enable_stream_processing"] = 1;
  params.target_specific_params["enable_dataflow_optimization"] = 1;
  
  return params;
}

SchedulingAlgorithm FPGASchedulingStrategy::getPrimaryAlgorithm() const {
  return SchedulingAlgorithm::FEAUTRIER; // Feautrier good for dataflow
}

std::vector<OptimizationTechnique> FPGASchedulingStrategy::getOptimizationTechniques() const {
  return {
    OptimizationTechnique::TILING,
    OptimizationTechnique::FUSION,
    OptimizationTechnique::UNROLLING,
    OptimizationTechnique::SKEWING,
    OptimizationTechnique::PRIVATIZATION
  };
}

double FPGASchedulingStrategy::calculatePriority(const target::TargetCharacteristics& target) const {
  if (target.type != target::TargetType::FPGA) return 0.0;
  
  // Lower priority due to complexity and constraints
  double priority = 6.0;
  
  // Boost for more compute units (more resources)
  priority += std::log2(std::max(1, target.compute_units)) * 0.2;
  
  return priority;
}

// Utility function implementations
std::string SchedulingUtils::algorithmToString(SchedulingAlgorithm algorithm) {
  switch (algorithm) {
    case SchedulingAlgorithm::ISL_SCHEDULER: return "ISL";
    case SchedulingAlgorithm::FEAUTRIER: return "Feautrier";
    case SchedulingAlgorithm::PLUTO: return "PLUTO";
    case SchedulingAlgorithm::PPCG_DEFAULT: return "PPCG";
    case SchedulingAlgorithm::CUSTOM: return "Custom";
  }
  return "Unknown";
}

std::string SchedulingUtils::techniqueToString(OptimizationTechnique technique) {
  switch (technique) {
    case OptimizationTechnique::TILING: return "Tiling";
    case OptimizationTechnique::FUSION: return "Fusion";
    case OptimizationTechnique::SKEWING: return "Skewing";
    case OptimizationTechnique::PARALLELIZATION: return "Parallelization";
    case OptimizationTechnique::VECTORIZATION: return "Vectorization";
    case OptimizationTechnique::UNROLLING: return "Unrolling";
    case OptimizationTechnique::INTERCHANGE: return "Interchange";
    case OptimizationTechnique::STRIP_MINING: return "Strip Mining";
    case OptimizationTechnique::PRIVATIZATION: return "Privatization";
  }
  return "Unknown";
}

bool SchedulingUtils::validateParameters(const SchedulingParameters& params,
                                       const target::TargetCharacteristics& target) {
  // Basic validation
  if (params.enable_tiling && params.tile_sizes.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: Tiling enabled but no tile sizes specified\n");
  }
  
  if (params.enable_unrolling && params.unroll_factors.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: Unrolling enabled but no unroll factors specified\n");
  }
  
  // Check parallel dimensions against target capabilities
  for (int dim : params.parallel_dimensions) {
    if (dim >= target.max_work_item_dimensions) {
      LLVM_DEBUG(llvm::dbgs() << "Error: Parallel dimension " << dim 
                              << " exceeds target capability " 
                              << target.max_work_item_dimensions << "\n");
      return false;
    }
  }
  
  // Check work group size constraints
  int total_work_items = 1;
  for (size_t i = 0; i < params.parallel_dimensions.size() && i < params.tile_sizes.size(); ++i) {
    total_work_items *= params.tile_sizes[i];
  }
  
  if (total_work_items > target.max_work_group_size) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: Total work items " << total_work_items
                            << " exceeds target max work group size " 
                            << target.max_work_group_size << "\n");
  }
  
  return true;
}

SchedulingParameters SchedulingUtils::mergeParameters(const SchedulingParameters& base,
                                                    const SchedulingParameters& override) {
  SchedulingParameters merged = base;
  
  // Override basic flags
  if (!override.tile_sizes.empty()) {
    merged.tile_sizes = override.tile_sizes;
    merged.enable_tiling = override.enable_tiling;
  }
  
  if (!override.parallel_dimensions.empty()) {
    merged.parallel_dimensions = override.parallel_dimensions;
  }
  
  if (!override.unroll_factors.empty()) {
    merged.unroll_factors = override.unroll_factors;
    merged.enable_unrolling = override.enable_unrolling;
  }
  
  if (!override.skewing_factors.empty()) {
    merged.skewing_factors = override.skewing_factors;
    merged.enable_skewing = override.enable_skewing;
  }
  
  // Override other parameters
  merged.max_parallel_depth = override.max_parallel_depth != 0 ? 
                             override.max_parallel_depth : merged.max_parallel_depth;
  
  merged.fusion_strategy = override.fusion_strategy;
  
  // Merge target-specific parameters
  for (const auto& param : override.target_specific_params) {
    merged.target_specific_params[param.first] = param.second;
  }
  
  return merged;
}

std::vector<int> SchedulingUtils::calculateOptimalTileSizes(
    const target::TargetCharacteristics& target,
    const std::vector<int>& loop_bounds,
    int data_element_size) {
  
  std::vector<int> tile_sizes;
  
  // Find appropriate memory level for tiling
  size_t target_memory_size = 32 * 1024; // Default L1 size
  
  for (const auto& mem : target.memory_hierarchy) {
    if (mem.level == target::MemoryLevel::LOCAL || 
        mem.level == target::MemoryLevel::SHARED) {
      target_memory_size = mem.size_bytes;
      break;
    }
  }
  
  // Calculate working set size and tile accordingly
  int num_arrays = 3; // Typical for compute kernels (A, B, C)
  size_t available_memory = target_memory_size / num_arrays;
  
  if (loop_bounds.size() == 1) {
    int tile_size = static_cast<int>(available_memory / data_element_size);
    tile_size = std::min(tile_size, loop_bounds[0]);
    tile_sizes.push_back(std::max(1, tile_size));
  } else if (loop_bounds.size() == 2) {
    int tile_size = static_cast<int>(std::sqrt(available_memory / data_element_size));
    for (int i = 0; i < 2; ++i) {
      int actual_size = std::min(tile_size, loop_bounds[i]);
      tile_sizes.push_back(std::max(1, actual_size));
    }
  } else if (loop_bounds.size() == 3) {
    int tile_size = static_cast<int>(std::cbrt(available_memory / data_element_size));
    for (int i = 0; i < 3; ++i) {
      int actual_size = std::min(tile_size, loop_bounds[i]);
      tile_sizes.push_back(std::max(1, actual_size));
    }
  } else {
    // General case: equal division
    double root = std::pow(available_memory / data_element_size, 1.0 / loop_bounds.size());
    int tile_size = static_cast<int>(root);
    for (size_t i = 0; i < loop_bounds.size(); ++i) {
      int actual_size = std::min(tile_size, loop_bounds[i]);
      tile_sizes.push_back(std::max(1, actual_size));
    }
  }
  
  return tile_sizes;
}

} // namespace scheduling
} // namespace autopoly
