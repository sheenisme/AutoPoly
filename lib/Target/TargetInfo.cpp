//===- TargetInfo.cpp - Target Information Implementation -----------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file implements target hardware detection and characterization
// for the AutoPoly polyhedral scheduling framework.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Target/TargetInfo.h"
#include "AutoPoly/Config.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <thread>

#define DEBUG_TYPE "target-info"

namespace autopoly {
namespace target {

std::string TargetUtils::targetTypeToString(TargetType type) {
  switch (type) {
    case TargetType::CPU: return "CPU";
    case TargetType::GPU: return "GPU";
    case TargetType::OPENCL: return "OpenCL";
    case TargetType::FPGA: return "FPGA";
    case TargetType::CGRA: return "CGRA";
    case TargetType::NPU: return "NPU";
    case TargetType::DPU: return "DPU";
    case TargetType::PIM: return "PIM";
    case TargetType::UNKNOWN: return "Unknown";
  }
  return "Unknown";
}

TargetType TargetUtils::stringToTargetType(const std::string& str) {
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
  
  if (lower_str == "cpu") return TargetType::CPU;
  if (lower_str == "gpu") return TargetType::GPU;
  if (lower_str == "opencl") return TargetType::OPENCL;
  if (lower_str == "fpga") return TargetType::FPGA;
  if (lower_str == "cgra") return TargetType::CGRA;
  if (lower_str == "npu") return TargetType::NPU;
  if (lower_str == "dpu") return TargetType::DPU;
  if (lower_str == "pim") return TargetType::PIM;
  
  return TargetType::UNKNOWN;
}

std::vector<int> TargetUtils::getRecommendedTileSizes(
    const TargetCharacteristics& target, int num_dimensions) {
  
  std::vector<int> tile_sizes;
  
  switch (target.type) {
    case TargetType::CPU: {
      // CPU cache-friendly tiling
      // Typically smaller tiles to fit in L1/L2 cache
      int base_size = 32;
      if (!target.memory_hierarchy.empty()) {
        // Use L1 cache size as reference
        auto l1_it = std::find_if(target.memory_hierarchy.begin(), 
                                  target.memory_hierarchy.end(),
                                  [](const auto& mem) { 
                                    return mem.level == MemoryLevel::LOCAL; 
                                  });
        if (l1_it != target.memory_hierarchy.end()) {
          // Estimate tile size based on cache size
          base_size = static_cast<int>(std::sqrt(l1_it->size_bytes / (4 * num_dimensions)));
          base_size = std::max(8, std::min(128, base_size));
        }
      }
      
      for (int i = 0; i < num_dimensions; ++i) {
        tile_sizes.push_back(base_size);
      }
      break;
    }
    
    case TargetType::GPU: {
      // GPU thread block-friendly tiling
      // Larger tiles to maximize parallelism
      int base_size = target.max_work_group_size > 0 ? 
                     static_cast<int>(std::sqrt(target.max_work_group_size)) : 16;
      
      for (int i = 0; i < num_dimensions; ++i) {
        if (i < target.max_work_item_sizes.size()) {
          tile_sizes.push_back(std::min(base_size, target.max_work_item_sizes[i]));
        } else {
          tile_sizes.push_back(base_size);
        }
      }
      break;
    }
    
    case TargetType::FPGA: {
      // FPGA resource-aware tiling
      // Moderate tile sizes for resource utilization
      int base_size = 16; // Conservative for resource constraints
      
      for (int i = 0; i < num_dimensions; ++i) {
        tile_sizes.push_back(base_size);
      }
      break;
    }
    
    default: {
      // Default conservative tiling
      for (int i = 0; i < num_dimensions; ++i) {
        tile_sizes.push_back(32);
      }
      break;
    }
  }
  
  return tile_sizes;
}

std::vector<int> TargetUtils::getRecommendedUnrollFactors(
    const TargetCharacteristics& target, int num_loops) {
  
  std::vector<int> unroll_factors;
  
  int base_factor = 1;
  switch (target.type) {
    case TargetType::CPU:
      base_factor = target.supports_vectorization ? 4 : 2;
      break;
    case TargetType::GPU:
      base_factor = 1; // Rely on hardware parallelism
      break;
    case TargetType::FPGA:
      base_factor = 8; // High unrolling for pipeline efficiency
      break;
    default:
      base_factor = 2;
      break;
  }
  
  for (int i = 0; i < num_loops; ++i) {
    unroll_factors.push_back(base_factor);
  }
  
  return unroll_factors;
}

std::map<MemoryLevel, int> TargetUtils::calculateMemoryParameters(
    const TargetCharacteristics& target) {
  
  std::map<MemoryLevel, int> params;
  
  for (const auto& mem_info : target.memory_hierarchy) {
    switch (mem_info.level) {
      case MemoryLevel::REGISTER:
        params[mem_info.level] = 1; // Prefetch distance
        break;
      case MemoryLevel::LOCAL:
        params[mem_info.level] = 4; // Cache line prefetch
        break;
      case MemoryLevel::SHARED:
        params[mem_info.level] = 16; // Shared memory blocks
        break;
      case MemoryLevel::GLOBAL:
        params[mem_info.level] = 64; // Global memory prefetch
        break;
    }
  }
  
  return params;
}

/// Base implementation of target detector
class BaseTargetDetector : public TargetDetector {
public:
  BaseTargetDetector() = default;
  virtual ~BaseTargetDetector() = default;

  std::vector<TargetCharacteristics> detectTargets() override {
    std::vector<TargetCharacteristics> targets;
    
    // Always detect CPU
    if (SUPPORT_CPU) {
      targets.push_back(detectCPU());
    }
    
    // Try to detect GPU
    if (SUPPORT_GPU) {
      auto gpu_targets = detectGPUs();
      targets.insert(targets.end(), gpu_targets.begin(), gpu_targets.end());
    }
    
    // Try to detect OpenCL devices
    if (SUPPORT_OPENCL) {
      auto opencl_targets = detectOpenCLDevices();
      targets.insert(targets.end(), opencl_targets.begin(), opencl_targets.end());
    }
    
    // Other targets would require specific detection logic
    if (SUPPORT_FPGA) {
      auto fpga_targets = detectFPGAs();
      targets.insert(targets.end(), fpga_targets.begin(), fpga_targets.end());
    }
    
    return targets;
  }

  TargetCharacteristics getDefaultTarget() override {
    auto targets = detectTargets();
    
    if (targets.empty()) {
      return createFallbackTarget();
    }
    
    // Prefer GPU > CPU > others
    for (const auto& target : targets) {
      if (target.type == TargetType::GPU) return target;
    }
    
    for (const auto& target : targets) {
      if (target.type == TargetType::CPU) return target;
    }
    
    return targets[0];
  }

  bool isTargetAvailable(TargetType type) override {
    auto targets = detectTargets();
    
    for (const auto& target : targets) {
      if (target.type == type) return true;
    }
    
    return false;
  }

  TargetCharacteristics getTargetByName(const std::string& name) override {
    auto targets = detectTargets();
    
    for (const auto& target : targets) {
      if (target.name == name) return target;
    }
    
    // Return default if not found
    return getDefaultTarget();
  }

private:
  TargetCharacteristics detectCPU() {
    TargetCharacteristics cpu;
    cpu.type = TargetType::CPU;
    cpu.name = "Generic CPU";
    cpu.vendor = "Unknown";
    
    // Try to detect CPU characteristics
    cpu.compute_units = std::thread::hardware_concurrency();
    cpu.max_work_group_size = cpu.compute_units;
    cpu.max_work_item_dimensions = 3;
    cpu.max_work_item_sizes = {1024, 1024, 1024};
    
    // Default memory hierarchy for CPU
    TargetCharacteristics::MemoryInfo l1_cache;
    l1_cache.level = MemoryLevel::LOCAL;
    l1_cache.size_bytes = 32 * 1024; // 32KB L1
    l1_cache.bandwidth_gb_per_s = 1000; // Very high for L1
    l1_cache.latency_cycles = 1;
    cpu.memory_hierarchy.push_back(l1_cache);
    
    TargetCharacteristics::MemoryInfo l2_cache;
    l2_cache.level = MemoryLevel::SHARED;
    l2_cache.size_bytes = 256 * 1024; // 256KB L2
    l2_cache.bandwidth_gb_per_s = 500;
    l2_cache.latency_cycles = 10;
    cpu.memory_hierarchy.push_back(l2_cache);
    
    TargetCharacteristics::MemoryInfo main_mem;
    main_mem.level = MemoryLevel::GLOBAL;
    main_mem.size_bytes = 8ULL * 1024 * 1024 * 1024; // 8GB
    main_mem.bandwidth_gb_per_s = 50;
    main_mem.latency_cycles = 300;
    cpu.memory_hierarchy.push_back(main_mem);
    
    // CPU capabilities
    cpu.supports_double_precision = true;
    cpu.supports_atomic_operations = true;
    cpu.supports_vectorization = true;
    cpu.supports_local_memory = true;
    
    cpu.peak_compute_throughput = cpu.compute_units * 2.5; // GHz * cores
    cpu.memory_coalescing_factor = 0.8;
    
    return cpu;
  }

  std::vector<TargetCharacteristics> detectGPUs() {
    std::vector<TargetCharacteristics> gpus;
    
    // Simplified GPU detection - in practice would use CUDA/OpenCL APIs
    TargetCharacteristics gpu;
    gpu.type = TargetType::GPU;
    gpu.name = "Generic GPU";
    gpu.vendor = "Unknown";
    
    gpu.compute_units = 1024; // Simplified
    gpu.max_work_group_size = 1024;
    gpu.max_work_item_dimensions = 3;
    gpu.max_work_item_sizes = {1024, 1024, 64};
    
    // GPU memory hierarchy
    TargetCharacteristics::MemoryInfo shared_mem;
    shared_mem.level = MemoryLevel::SHARED;
    shared_mem.size_bytes = 48 * 1024; // 48KB shared memory
    shared_mem.bandwidth_gb_per_s = 8000;
    shared_mem.latency_cycles = 1;
    gpu.memory_hierarchy.push_back(shared_mem);
    
    TargetCharacteristics::MemoryInfo global_mem;
    global_mem.level = MemoryLevel::GLOBAL;
    global_mem.size_bytes = 8ULL * 1024 * 1024 * 1024; // 8GB
    global_mem.bandwidth_gb_per_s = 900;
    global_mem.latency_cycles = 400;
    gpu.memory_hierarchy.push_back(global_mem);
    
    gpu.supports_double_precision = true;
    gpu.supports_atomic_operations = true;
    gpu.supports_vectorization = true;
    gpu.supports_local_memory = true;
    
    gpu.peak_compute_throughput = 10000; // 10 TFLOPS
    gpu.memory_coalescing_factor = 0.95;
    
    gpus.push_back(gpu);
    return gpus;
  }

  std::vector<TargetCharacteristics> detectOpenCLDevices() {
    // Would use OpenCL APIs in practice
    return {}; // Simplified for now
  }

  std::vector<TargetCharacteristics> detectFPGAs() {
    // Would use FPGA-specific detection in practice
    return {}; // Simplified for now
  }

  TargetCharacteristics createFallbackTarget() {
    TargetCharacteristics fallback;
    fallback.type = TargetType::CPU;
    fallback.name = "Fallback CPU";
    fallback.vendor = "Generic";
    fallback.compute_units = 1;
    fallback.max_work_group_size = 1;
    fallback.max_work_item_dimensions = 1;
    fallback.max_work_item_sizes = {1};
    
    TargetCharacteristics::MemoryInfo main_mem;
    main_mem.level = MemoryLevel::GLOBAL;
    main_mem.size_bytes = 1024 * 1024 * 1024; // 1GB
    main_mem.bandwidth_gb_per_s = 10;
    main_mem.latency_cycles = 100;
    fallback.memory_hierarchy.push_back(main_mem);
    
    fallback.supports_double_precision = true;
    fallback.supports_atomic_operations = false;
    fallback.supports_vectorization = false;
    fallback.supports_local_memory = false;
    
    fallback.peak_compute_throughput = 1.0;
    fallback.memory_coalescing_factor = 0.1;
    
    return fallback;
  }
};

/// Mock target detector for testing
class MockTargetDetector : public TargetDetector {
private:
  std::vector<TargetCharacteristics> mock_targets_;

public:
  explicit MockTargetDetector(const std::vector<TargetCharacteristics>& targets)
      : mock_targets_(targets) {}

  std::vector<TargetCharacteristics> detectTargets() override {
    return mock_targets_;
  }

  TargetCharacteristics getDefaultTarget() override {
    if (mock_targets_.empty()) {
      TargetCharacteristics fallback;
      fallback.type = TargetType::UNKNOWN;
      return fallback;
    }
    return mock_targets_[0];
  }

  bool isTargetAvailable(TargetType type) override {
    for (const auto& target : mock_targets_) {
      if (target.type == type) return true;
    }
    return false;
  }

  TargetCharacteristics getTargetByName(const std::string& name) override {
    for (const auto& target : mock_targets_) {
      if (target.name == name) return target;
    }
    return getDefaultTarget();
  }
};

std::unique_ptr<TargetDetector> TargetDetectorFactory::createDetector() {
  return std::make_unique<BaseTargetDetector>();
}

std::unique_ptr<TargetDetector> TargetDetectorFactory::createMockDetector(
    const std::vector<TargetCharacteristics>& targets) {
  return std::make_unique<MockTargetDetector>(targets);
}

} // namespace target
} // namespace autopoly
