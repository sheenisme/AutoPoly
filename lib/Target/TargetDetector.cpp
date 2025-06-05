//===- TargetDetector.cpp - Target Hardware Detection -----*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements the target hardware detection and characterization
// functionality for automatic target-specific optimization in the AutoPoly
// polyhedral scheduling framework.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Target/TargetInfo.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <memory>
#include <thread>
#include <fstream>
#include <sstream>

#define DEBUG_TYPE "target-detection"

namespace autopoly {
namespace target {

/// Concrete implementation of target detector
class SystemTargetDetector : public TargetDetector {
public:
  SystemTargetDetector();
  ~SystemTargetDetector() override = default;

  std::vector<TargetCharacteristics> detectTargets() override;
  TargetCharacteristics getDefaultTarget() override;
  bool isTargetAvailable(TargetType type) override;
  TargetCharacteristics getTargetByName(const std::string& name) override;

private:
  std::vector<TargetCharacteristics> detected_targets_;
  
  // Detection methods for different target types
  std::vector<TargetCharacteristics> detectCPUTargets();
  std::vector<TargetCharacteristics> detectGPUTargets();
  std::vector<TargetCharacteristics> detectOpenCLTargets();
  std::vector<TargetCharacteristics> detectFPGATargets();
  std::vector<TargetCharacteristics> detectNPUTargets();
  std::vector<TargetCharacteristics> detectPIMTargets();
  
  // Helper methods
  TargetCharacteristics createCPUCharacteristics();
  TargetCharacteristics createDefaultGPUCharacteristics();
  TargetCharacteristics createDefaultFPGACharacteristics();
  TargetCharacteristics createDefaultNPUCharacteristics();
  
  // System information gathering
  int getCPUCoreCount();
  size_t getSystemMemorySize();
  std::string getCPUVendor();
  std::string getCPUModel();
  bool hasAVXSupport();
  bool hasGPUDevice();
  
  void characterizeTarget(TargetCharacteristics& target);
};

/// Mock target detector for testing
class MockTargetDetector : public TargetDetector {
public:
  explicit MockTargetDetector(const std::vector<TargetCharacteristics>& targets)
      : mock_targets_(targets) {}
  
  std::vector<TargetCharacteristics> detectTargets() override {
    return mock_targets_;
  }
  
  TargetCharacteristics getDefaultTarget() override {
    if (!mock_targets_.empty()) {
      return mock_targets_[0];
    }
    return createFallbackTarget();
  }
  
  bool isTargetAvailable(TargetType type) override {
    for (const auto& target : mock_targets_) {
      if (target.type == type) {
        return true;
      }
    }
    return false;
  }
  
  TargetCharacteristics getTargetByName(const std::string& name) override {
    for (const auto& target : mock_targets_) {
      if (target.name == name) {
        return target;
      }
    }
    return createFallbackTarget();
  }

private:
  std::vector<TargetCharacteristics> mock_targets_;
  
  TargetCharacteristics createFallbackTarget() {
    TargetCharacteristics fallback;
    fallback.type = TargetType::CPU;
    fallback.name = "fallback_cpu";
    fallback.vendor = "unknown";
    fallback.compute_units = 1;
    fallback.max_work_group_size = 1;
    fallback.max_work_item_dimensions = 3;
    fallback.max_work_item_sizes = {1, 1, 1};
    fallback.supports_double_precision = true;
    fallback.supports_atomic_operations = true;
    fallback.supports_vectorization = false;
    fallback.supports_local_memory = false;
    fallback.peak_compute_throughput = 1.0;
    fallback.memory_coalescing_factor = 1.0;
    return fallback;
  }
};

//===----------------------------------------------------------------------===//
// SystemTargetDetector Implementation
//===----------------------------------------------------------------------===//

SystemTargetDetector::SystemTargetDetector() {
  // Perform initial detection
  detectTargets();
}

std::vector<TargetCharacteristics> SystemTargetDetector::detectTargets() {
  detected_targets_.clear();
  
  LLVM_DEBUG(llvm::dbgs() << "Starting target detection...\n");
  
  // Detect different target types
  auto cpu_targets = detectCPUTargets();
  auto gpu_targets = detectGPUTargets();
  auto opencl_targets = detectOpenCLTargets();
  auto fpga_targets = detectFPGATargets();
  auto npu_targets = detectNPUTargets();
  auto pim_targets = detectPIMTargets();
  
  // Combine all detected targets
  detected_targets_.insert(detected_targets_.end(), cpu_targets.begin(), cpu_targets.end());
  detected_targets_.insert(detected_targets_.end(), gpu_targets.begin(), gpu_targets.end());
  detected_targets_.insert(detected_targets_.end(), opencl_targets.begin(), opencl_targets.end());
  detected_targets_.insert(detected_targets_.end(), fpga_targets.begin(), fpga_targets.end());
  detected_targets_.insert(detected_targets_.end(), npu_targets.begin(), npu_targets.end());
  detected_targets_.insert(detected_targets_.end(), pim_targets.begin(), pim_targets.end());
  
  // If no targets detected, create a fallback CPU target
  if (detected_targets_.empty()) {
    detected_targets_.push_back(createCPUCharacteristics());
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Target detection complete. Found " 
                         << detected_targets_.size() << " targets\n");
  
  return detected_targets_;
}

TargetCharacteristics SystemTargetDetector::getDefaultTarget() {
  if (detected_targets_.empty()) {
    detectTargets();
  }
  
  // Priority order: GPU > NPU > CPU > others
  std::vector<TargetType> priority_order = {
    TargetType::GPU, TargetType::NPU, TargetType::CPU, 
    TargetType::OPENCL, TargetType::FPGA, TargetType::CGRA, 
    TargetType::DPU, TargetType::PIM
  };
  
  for (auto preferred_type : priority_order) {
    for (const auto& target : detected_targets_) {
      if (target.type == preferred_type) {
        LLVM_DEBUG(llvm::dbgs() << "Selected default target: " << target.name 
                               << " (type: " << TargetUtils::targetTypeToString(target.type) << ")\n");
        return target;
      }
    }
  }
  
  // Fallback to first available target
  if (!detected_targets_.empty()) {
    return detected_targets_[0];
  }
  
  // Last resort: create a basic CPU target
  return createCPUCharacteristics();
}

bool SystemTargetDetector::isTargetAvailable(TargetType type) {
  if (detected_targets_.empty()) {
    detectTargets();
  }
  
  for (const auto& target : detected_targets_) {
    if (target.type == type) {
      return true;
    }
  }
  return false;
}

TargetCharacteristics SystemTargetDetector::getTargetByName(const std::string& name) {
  if (detected_targets_.empty()) {
    detectTargets();
  }
  
  for (const auto& target : detected_targets_) {
    if (target.name == name) {
      return target;
    }
  }
  
  // If not found, return default target
  return getDefaultTarget();
}

std::vector<TargetCharacteristics> SystemTargetDetector::detectCPUTargets() {
  std::vector<TargetCharacteristics> cpu_targets;
  
  TargetCharacteristics cpu = createCPUCharacteristics();
  characterizeTarget(cpu);
  cpu_targets.push_back(cpu);
  
  LLVM_DEBUG(llvm::dbgs() << "Detected CPU target: " << cpu.name 
                         << " with " << cpu.compute_units << " cores\n");
  
  return cpu_targets;
}

std::vector<TargetCharacteristics> SystemTargetDetector::detectGPUTargets() {
  std::vector<TargetCharacteristics> gpu_targets;
  
  // Simple GPU detection - in practice would use CUDA/ROCm APIs
  if (hasGPUDevice()) {
    TargetCharacteristics gpu = createDefaultGPUCharacteristics();
    characterizeTarget(gpu);
    gpu_targets.push_back(gpu);
    
    LLVM_DEBUG(llvm::dbgs() << "Detected GPU target: " << gpu.name << "\n");
  }
  
  return gpu_targets;
}

std::vector<TargetCharacteristics> SystemTargetDetector::detectOpenCLTargets() {
  std::vector<TargetCharacteristics> opencl_targets;
  
  // OpenCL detection would use OpenCL APIs
  // For now, return empty vector
  
  return opencl_targets;
}

std::vector<TargetCharacteristics> SystemTargetDetector::detectFPGATargets() {
  std::vector<TargetCharacteristics> fpga_targets;
  
  // FPGA detection would check for specific vendors/frameworks
  // For demonstration, create a default FPGA target if environment suggests it
  const char* fpga_env = std::getenv("AUTOPOLY_FPGA_TARGET");
  if (fpga_env) {
    TargetCharacteristics fpga = createDefaultFPGACharacteristics();
    fpga.name = fpga_env;
    characterizeTarget(fpga);
    fpga_targets.push_back(fpga);
    
    LLVM_DEBUG(llvm::dbgs() << "Detected FPGA target: " << fpga.name << "\n");
  }
  
  return fpga_targets;
}

std::vector<TargetCharacteristics> SystemTargetDetector::detectNPUTargets() {
  std::vector<TargetCharacteristics> npu_targets;
  
  // NPU detection would check for specific neural processing units
  const char* npu_env = std::getenv("AUTOPOLY_NPU_TARGET");
  if (npu_env) {
    TargetCharacteristics npu = createDefaultNPUCharacteristics();
    npu.name = npu_env;
    characterizeTarget(npu);
    npu_targets.push_back(npu);
    
    LLVM_DEBUG(llvm::dbgs() << "Detected NPU target: " << npu.name << "\n");
  }
  
  return npu_targets;
}

std::vector<TargetCharacteristics> SystemTargetDetector::detectPIMTargets() {
  std::vector<TargetCharacteristics> pim_targets;
  
  // PIM detection would check for processing-in-memory devices
  // For now, return empty vector
  
  return pim_targets;
}

TargetCharacteristics SystemTargetDetector::createCPUCharacteristics() {
  TargetCharacteristics cpu;
  cpu.type = TargetType::CPU;
  cpu.name = getCPUModel();
  cpu.vendor = getCPUVendor();
  cpu.compute_units = getCPUCoreCount();
  cpu.max_work_group_size = cpu.compute_units;
  cpu.max_work_item_dimensions = 3;
  cpu.max_work_item_sizes = {cpu.compute_units, 1, 1};
  
  // Memory hierarchy for CPU
  TargetCharacteristics::MemoryInfo l1_cache;
  l1_cache.level = MemoryLevel::REGISTER;
  l1_cache.size_bytes = 32 * 1024; // 32KB L1 cache
  l1_cache.bandwidth_gb_per_s = 500;
  l1_cache.latency_cycles = 1;
  cpu.memory_hierarchy.push_back(l1_cache);
  
  TargetCharacteristics::MemoryInfo l2_cache;
  l2_cache.level = MemoryLevel::LOCAL;
  l2_cache.size_bytes = 256 * 1024; // 256KB L2 cache
  l2_cache.bandwidth_gb_per_s = 200;
  l2_cache.latency_cycles = 10;
  cpu.memory_hierarchy.push_back(l2_cache);
  
  TargetCharacteristics::MemoryInfo main_memory;
  main_memory.level = MemoryLevel::GLOBAL;
  main_memory.size_bytes = getSystemMemorySize();
  main_memory.bandwidth_gb_per_s = 50; // Typical DDR4 bandwidth
  main_memory.latency_cycles = 300;
  cpu.memory_hierarchy.push_back(main_memory);
  
  // CPU capabilities
  cpu.supports_double_precision = true;
  cpu.supports_atomic_operations = true;
  cpu.supports_vectorization = hasAVXSupport();
  cpu.supports_local_memory = true;
  
  // Performance characteristics
  cpu.peak_compute_throughput = cpu.compute_units * 2.0; // Estimate: 2 GFLOPS per core
  cpu.memory_coalescing_factor = 1.0; // CPUs don't have strict coalescing requirements
  
  // Scheduling parameters
  cpu.scheduling_parameters["preferred_tile_size"] = 32;
  cpu.scheduling_parameters["max_unroll_factor"] = 8;
  cpu.scheduling_parameters["preferred_parallel_depth"] = 2;
  
  return cpu;
}

TargetCharacteristics SystemTargetDetector::createDefaultGPUCharacteristics() {
  TargetCharacteristics gpu;
  gpu.type = TargetType::GPU;
  gpu.name = "generic_gpu";
  gpu.vendor = "unknown";
  gpu.compute_units = 32; // Typical mid-range GPU
  gpu.max_work_group_size = 1024;
  gpu.max_work_item_dimensions = 3;
  gpu.max_work_item_sizes = {1024, 1024, 64};
  
  // GPU memory hierarchy
  TargetCharacteristics::MemoryInfo registers;
  registers.level = MemoryLevel::REGISTER;
  registers.size_bytes = 64 * 1024; // 64KB registers per SM
  registers.bandwidth_gb_per_s = 2000;
  registers.latency_cycles = 1;
  gpu.memory_hierarchy.push_back(registers);
  
  TargetCharacteristics::MemoryInfo shared_mem;
  shared_mem.level = MemoryLevel::SHARED;
  shared_mem.size_bytes = 48 * 1024; // 48KB shared memory per SM
  shared_mem.bandwidth_gb_per_s = 1500;
  shared_mem.latency_cycles = 1;
  gpu.memory_hierarchy.push_back(shared_mem);
  
  TargetCharacteristics::MemoryInfo global_mem;
  global_mem.level = MemoryLevel::GLOBAL;
  global_mem.size_bytes = 8ULL * 1024 * 1024 * 1024; // 8GB VRAM
  global_mem.bandwidth_gb_per_s = 400; // High-bandwidth memory
  global_mem.latency_cycles = 400;
  gpu.memory_hierarchy.push_back(global_mem);
  
  // GPU capabilities
  gpu.supports_double_precision = false; // Many GPUs have limited FP64
  gpu.supports_atomic_operations = true;
  gpu.supports_vectorization = true;
  gpu.supports_local_memory = true;
  
  // Performance characteristics
  gpu.peak_compute_throughput = gpu.compute_units * 100.0; // Higher throughput per unit
  gpu.memory_coalescing_factor = 8.0; // GPUs benefit significantly from coalescing
  
  // GPU-specific scheduling parameters
  gpu.scheduling_parameters["preferred_tile_size"] = 16;
  gpu.scheduling_parameters["max_unroll_factor"] = 4;
  gpu.scheduling_parameters["preferred_parallel_depth"] = 3;
  gpu.scheduling_parameters["warp_size"] = 32;
  gpu.scheduling_parameters["max_threads_per_block"] = 1024;
  
  return gpu;
}

TargetCharacteristics SystemTargetDetector::createDefaultFPGACharacteristics() {
  TargetCharacteristics fpga;
  fpga.type = TargetType::FPGA;
  fpga.name = "generic_fpga";
  fpga.vendor = "xilinx"; // Default to Xilinx
  fpga.compute_units = 16; // Configurable processing elements
  fpga.max_work_group_size = 256;
  fpga.max_work_item_dimensions = 3;
  fpga.max_work_item_sizes = {256, 256, 1};
  
  // FPGA memory hierarchy (simplified)
  TargetCharacteristics::MemoryInfo block_ram;
  block_ram.level = MemoryLevel::LOCAL;
  block_ram.size_bytes = 2 * 1024 * 1024; // 2MB block RAM
  block_ram.bandwidth_gb_per_s = 100;
  block_ram.latency_cycles = 2;
  fpga.memory_hierarchy.push_back(block_ram);
  
  TargetCharacteristics::MemoryInfo ddr_mem;
  ddr_mem.level = MemoryLevel::GLOBAL;
  ddr_mem.size_bytes = 4ULL * 1024 * 1024 * 1024; // 4GB DDR
  ddr_mem.bandwidth_gb_per_s = 25;
  ddr_mem.latency_cycles = 100;
  fpga.memory_hierarchy.push_back(ddr_mem);
  
  // FPGA capabilities
  fpga.supports_double_precision = true; // Configurable precision
  fpga.supports_atomic_operations = false; // Limited atomic support
  fpga.supports_vectorization = true; // Custom vector units
  fpga.supports_local_memory = true;
  
  // Performance characteristics
  fpga.peak_compute_throughput = fpga.compute_units * 5.0; // Lower frequency, but efficient
  fpga.memory_coalescing_factor = 2.0; // Some benefit from structured access
  
  // FPGA-specific scheduling parameters
  fpga.scheduling_parameters["preferred_tile_size"] = 64;
  fpga.scheduling_parameters["max_unroll_factor"] = 16; // FPGAs benefit from unrolling
  fpga.scheduling_parameters["preferred_parallel_depth"] = 1;
  fpga.scheduling_parameters["pipeline_stages"] = 8;
  
  return fpga;
}

TargetCharacteristics SystemTargetDetector::createDefaultNPUCharacteristics() {
  TargetCharacteristics npu;
  npu.type = TargetType::NPU;
  npu.name = "generic_npu";
  npu.vendor = "unknown";
  npu.compute_units = 8; // Neural processing cores
  npu.max_work_group_size = 128;
  npu.max_work_item_dimensions = 2; // Typically 2D for neural networks
  npu.max_work_item_sizes = {128, 128, 1};
  
  // NPU memory hierarchy
  TargetCharacteristics::MemoryInfo neural_cache;
  neural_cache.level = MemoryLevel::LOCAL;
  neural_cache.size_bytes = 1024 * 1024; // 1MB neural cache
  neural_cache.bandwidth_gb_per_s = 200;
  neural_cache.latency_cycles = 5;
  npu.memory_hierarchy.push_back(neural_cache);
  
  TargetCharacteristics::MemoryInfo main_mem;
  main_mem.level = MemoryLevel::GLOBAL;
  main_mem.size_bytes = 2ULL * 1024 * 1024 * 1024; // 2GB dedicated memory
  main_mem.bandwidth_gb_per_s = 150;
  main_mem.latency_cycles = 50;
  npu.memory_hierarchy.push_back(main_mem);
  
  // NPU capabilities
  npu.supports_double_precision = false; // NPUs typically use lower precision
  npu.supports_atomic_operations = false;
  npu.supports_vectorization = true; // Specialized for vector operations
  npu.supports_local_memory = true;
  
  // Performance characteristics
  npu.peak_compute_throughput = npu.compute_units * 50.0; // High throughput for neural ops
  npu.memory_coalescing_factor = 4.0; // Benefits from structured data access
  
  // NPU-specific scheduling parameters
  npu.scheduling_parameters["preferred_tile_size"] = 8;
  npu.scheduling_parameters["max_unroll_factor"] = 4;
  npu.scheduling_parameters["preferred_parallel_depth"] = 2;
  npu.scheduling_parameters["tensor_block_size"] = 16;
  
  return npu;
}

void SystemTargetDetector::characterizeTarget(TargetCharacteristics& target) {
  // Additional characterization based on target type
  switch (target.type) {
    case TargetType::CPU:
      // CPU-specific optimizations
      if (target.supports_vectorization) {
        target.scheduling_parameters["vectorization_factor"] = 8;
      }
      break;
      
    case TargetType::GPU:
      // GPU-specific optimizations
      target.scheduling_parameters["coalescing_factor"] = static_cast<int>(target.memory_coalescing_factor);
      break;
      
    case TargetType::FPGA:
      // FPGA-specific optimizations
      target.scheduling_parameters["resource_sharing"] = 1;
      break;
      
    default:
      break;
  }
}

int SystemTargetDetector::getCPUCoreCount() {
  return std::max(1U, std::thread::hardware_concurrency());
}

size_t SystemTargetDetector::getSystemMemorySize() {
  // Try to get system memory size (simplified)
  size_t default_size = 8ULL * 1024 * 1024 * 1024; // 8GB default
  
#ifdef __linux__
  std::ifstream meminfo("/proc/meminfo");
  if (meminfo.is_open()) {
    std::string line;
    while (std::getline(meminfo, line)) {
      if (line.find("MemTotal:") == 0) {
        std::istringstream iss(line);
        std::string label, unit;
        size_t size_kb;
        iss >> label >> size_kb >> unit;
        return size_kb * 1024; // Convert KB to bytes
      }
    }
  }
#endif
  
  return default_size;
}

std::string SystemTargetDetector::getCPUVendor() {
  // Simplified CPU vendor detection
#ifdef __linux__
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo.is_open()) {
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("vendor_id") != std::string::npos) {
        size_t pos = line.find(":");
        if (pos != std::string::npos) {
          std::string vendor = line.substr(pos + 1);
          // Trim whitespace
          vendor.erase(0, vendor.find_first_not_of(" \t"));
          vendor.erase(vendor.find_last_not_of(" \t") + 1);
          return vendor;
        }
      }
    }
  }
#endif
  
  return "unknown";
}

std::string SystemTargetDetector::getCPUModel() {
  // Simplified CPU model detection
#ifdef __linux__
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo.is_open()) {
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("model name") != std::string::npos) {
        size_t pos = line.find(":");
        if (pos != std::string::npos) {
          std::string model = line.substr(pos + 1);
          // Trim whitespace
          model.erase(0, model.find_first_not_of(" \t"));
          model.erase(model.find_last_not_of(" \t") + 1);
          return model;
        }
      }
    }
  }
#endif
  
  return "generic_cpu";
}

bool SystemTargetDetector::hasAVXSupport() {
  // Simplified AVX detection
#ifdef __linux__
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo.is_open()) {
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("flags") != std::string::npos) {
        return line.find("avx") != std::string::npos;
      }
    }
  }
#endif
  
  return false; // Conservative default
}

bool SystemTargetDetector::hasGPUDevice() {
  // Simple GPU detection - check for NVIDIA or AMD drivers
#ifdef __linux__
  std::ifstream nvidia_proc("/proc/driver/nvidia/version");
  if (nvidia_proc.is_open()) {
    return true;
  }
  
  // Check for AMD GPU
  std::ifstream amd_proc("/sys/class/drm/card0/device/vendor");
  if (amd_proc.is_open()) {
    std::string vendor_id;
    amd_proc >> vendor_id;
    if (vendor_id == "0x1002") { // AMD vendor ID
      return true;
    }
  }
#endif
  
  // Check environment variable for GPU simulation
  return std::getenv("AUTOPOLY_GPU_TARGET") != nullptr;
}

//===----------------------------------------------------------------------===//
// TargetDetectorFactory Implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<TargetDetector> TargetDetectorFactory::createDetector() {
  return std::make_unique<SystemTargetDetector>();
}

std::unique_ptr<TargetDetector> TargetDetectorFactory::createMockDetector(
    const std::vector<TargetCharacteristics>& targets) {
  return std::make_unique<MockTargetDetector>(targets);
}

//===----------------------------------------------------------------------===//
// TargetUtils Implementation
//===----------------------------------------------------------------------===//

std::string TargetUtils::targetTypeToString(TargetType type) {
  switch (type) {
    case TargetType::CPU: return "cpu";
    case TargetType::GPU: return "gpu";
    case TargetType::OPENCL: return "opencl";
    case TargetType::FPGA: return "fpga";
    case TargetType::CGRA: return "cgra";
    case TargetType::NPU: return "npu";
    case TargetType::DPU: return "dpu";
    case TargetType::PIM: return "pim";
    case TargetType::UNKNOWN: return "unknown";
  }
  return "unknown";
}

TargetType TargetUtils::stringToTargetType(const std::string& str) {
  if (str == "cpu") return TargetType::CPU;
  if (str == "gpu") return TargetType::GPU;
  if (str == "opencl") return TargetType::OPENCL;
  if (str == "fpga") return TargetType::FPGA;
  if (str == "cgra") return TargetType::CGRA;
  if (str == "npu") return TargetType::NPU;
  if (str == "dpu") return TargetType::DPU;
  if (str == "pim") return TargetType::PIM;
  return TargetType::UNKNOWN;
}

std::vector<int> TargetUtils::getRecommendedTileSizes(
    const TargetCharacteristics& target, int num_dimensions) {
  
  std::vector<int> tile_sizes(num_dimensions);
  
  int base_tile_size = 32; // Default
  
  // Get target-specific preferred tile size
  auto it = target.scheduling_parameters.find("preferred_tile_size");
  if (it != target.scheduling_parameters.end()) {
    base_tile_size = it->second;
  }
  
  // Adjust based on target type
  switch (target.type) {
    case TargetType::CPU:
      // CPU: moderate tile sizes for cache locality
      for (int i = 0; i < num_dimensions; ++i) {
        tile_sizes[i] = base_tile_size;
      }
      break;
      
    case TargetType::GPU:
      // GPU: smaller tiles for better parallelism
      for (int i = 0; i < num_dimensions; ++i) {
        tile_sizes[i] = std::max(8, base_tile_size / 2);
      }
      break;
      
    case TargetType::FPGA:
      // FPGA: larger tiles for resource efficiency
      for (int i = 0; i < num_dimensions; ++i) {
        tile_sizes[i] = base_tile_size * 2;
      }
      break;
      
    case TargetType::NPU:
      // NPU: small tiles optimized for neural operations
      for (int i = 0; i < num_dimensions; ++i) {
        tile_sizes[i] = std::max(4, base_tile_size / 4);
      }
      break;
      
    default:
      for (int i = 0; i < num_dimensions; ++i) {
        tile_sizes[i] = base_tile_size;
      }
      break;
  }
  
  return tile_sizes;
}

std::vector<int> TargetUtils::getRecommendedUnrollFactors(
    const TargetCharacteristics& target, int num_loops) {
  
  std::vector<int> unroll_factors(num_loops, 1);
  
  int max_unroll = 4; // Default
  
  // Get target-specific max unroll factor
  auto it = target.scheduling_parameters.find("max_unroll_factor");
  if (it != target.scheduling_parameters.end()) {
    max_unroll = it->second;
  }
  
  // Apply target-specific unrolling strategy
  switch (target.type) {
    case TargetType::CPU:
      // Moderate unrolling for CPUs
      for (int i = 0; i < num_loops; ++i) {
        unroll_factors[i] = std::min(max_unroll, 4);
      }
      break;
      
    case TargetType::GPU:
      // Limited unrolling for GPUs (relies on many threads)
      for (int i = 0; i < num_loops; ++i) {
        unroll_factors[i] = std::min(max_unroll, 2);
      }
      break;
      
    case TargetType::FPGA:
      // Aggressive unrolling for FPGAs
      for (int i = 0; i < num_loops; ++i) {
        unroll_factors[i] = max_unroll;
      }
      break;
      
    default:
      // Conservative default
      for (int i = 0; i < num_loops; ++i) {
        unroll_factors[i] = 2;
      }
      break;
  }
  
  return unroll_factors;
}

std::map<MemoryLevel, int> TargetUtils::calculateMemoryParameters(
    const TargetCharacteristics& target) {
  
  std::map<MemoryLevel, int> params;
  
  // Calculate parameters based on memory hierarchy
  for (const auto& mem_info : target.memory_hierarchy) {
    switch (mem_info.level) {
      case MemoryLevel::REGISTER:
        params[MemoryLevel::REGISTER] = static_cast<int>(mem_info.size_bytes / 1024); // Size in KB
        break;
        
      case MemoryLevel::LOCAL:
        params[MemoryLevel::LOCAL] = static_cast<int>(mem_info.size_bytes / 1024); // Size in KB
        break;
        
      case MemoryLevel::SHARED:
        params[MemoryLevel::SHARED] = static_cast<int>(mem_info.size_bytes / 1024); // Size in KB
        break;
        
      case MemoryLevel::GLOBAL:
        params[MemoryLevel::GLOBAL] = static_cast<int>(mem_info.bandwidth_gb_per_s); // Bandwidth
        break;
    }
  }
  
  return params;
}

} // namespace target
} // namespace autopoly
