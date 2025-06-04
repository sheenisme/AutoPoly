//===- TargetDetector.cpp - Advanced Target Detection ---------*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements advanced target detection mechanisms including
// runtime detection of CPU, GPU, and other accelerators.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Target/TargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "AutoPoly/Config.h"

#include <fstream>
#include <thread>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#define DEBUG_TYPE "target-detector"

namespace autopoly {
namespace target {

/// Advanced CPU detection with /proc filesystem parsing
class AdvancedCPUDetector {
public:
  static TargetCharacteristics detectCPU() {
    TargetCharacteristics cpu;
    cpu.type = TargetType::CPU;
    cpu.name = getCPUName();
    cpu.vendor = getCPUVendor();
    
    cpu.compute_units = std::thread::hardware_concurrency();
    cpu.max_work_group_size = cpu.compute_units;
    cpu.max_work_item_dimensions = 3;
    cpu.max_work_item_sizes = {1024, 1024, 1024};
    
    // Detect CPU capabilities
    detectCPUCapabilities(cpu);
    
    // Detect memory hierarchy
    detectMemoryHierarchy(cpu);
    
    // Performance characteristics
    cpu.peak_compute_throughput = estimateComputeThroughput(cpu);
    cpu.memory_coalescing_factor = 0.8; // Typical for CPU
    
    // Scheduling parameters
    cpu.scheduling_parameters["preferred_tile_size"] = 32;
    cpu.scheduling_parameters["cache_line_size"] = 64;
    cpu.scheduling_parameters["max_unroll_factor"] = cpu.supports_vectorization ? 8 : 4;
    
    return cpu;
  }

private:
  static std::string getCPUName() {
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("model name") != std::string::npos) {
        auto pos = line.find(":");
        if (pos != std::string::npos) {
          return line.substr(pos + 2);
        }
      }
    }
#endif
    return "Unknown CPU";
  }

  static std::string getCPUVendor() {
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("vendor_id") != std::string::npos) {
        auto pos = line.find(":");
        if (pos != std::string::npos) {
          std::string vendor = line.substr(pos + 2);
          if (vendor.find("Intel") != std::string::npos) return "Intel";
          if (vendor.find("AMD") != std::string::npos) return "AMD";
          if (vendor.find("ARM") != std::string::npos) return "ARM";
          return vendor;
        }
      }
    }
#endif
    return "Unknown";
  }

  static void detectCPUCapabilities(TargetCharacteristics& cpu) {
    cpu.supports_double_precision = true; // Almost all modern CPUs
    cpu.supports_atomic_operations = true;
    cpu.supports_local_memory = true;
    
    // Detect vectorization support
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("flags") != std::string::npos) {
        cpu.supports_vectorization = (line.find("sse") != std::string::npos ||
                                     line.find("avx") != std::string::npos ||
                                     line.find("neon") != std::string::npos);
        break;
      }
    }
#else
    cpu.supports_vectorization = true; // Assume modern CPU
#endif
  }

  static void detectMemoryHierarchy(TargetCharacteristics& cpu) {
    // Try to detect actual cache sizes
    auto l1_size = detectCacheSize("L1d");
    auto l2_size = detectCacheSize("L2");
    auto l3_size = detectCacheSize("L3");
    
    // L1 Data Cache
    TargetCharacteristics::MemoryInfo l1;
    l1.level = MemoryLevel::LOCAL;
    l1.size_bytes = l1_size > 0 ? l1_size : 32 * 1024; // Default 32KB
    l1.bandwidth_gb_per_s = 1000; // Very high for L1
    l1.latency_cycles = 1;
    cpu.memory_hierarchy.push_back(l1);
    
    // L2 Cache
    if (l2_size > 0 || l3_size == 0) { // Has L2 or no L3
      TargetCharacteristics::MemoryInfo l2;
      l2.level = MemoryLevel::SHARED;
      l2.size_bytes = l2_size > 0 ? l2_size : 256 * 1024; // Default 256KB
      l2.bandwidth_gb_per_s = 500;
      l2.latency_cycles = 10;
      cpu.memory_hierarchy.push_back(l2);
    }
    
    // L3 Cache (if present)
    if (l3_size > 0) {
      TargetCharacteristics::MemoryInfo l3;
      l3.level = MemoryLevel::SHARED;
      l3.size_bytes = l3_size;
      l3.bandwidth_gb_per_s = 200;
      l3.latency_cycles = 30;
      cpu.memory_hierarchy.push_back(l3);
    }
    
    // Main Memory
    TargetCharacteristics::MemoryInfo main_mem;
    main_mem.level = MemoryLevel::GLOBAL;
    main_mem.size_bytes = getTotalMemory();
    main_mem.bandwidth_gb_per_s = estimateMemoryBandwidth();
    main_mem.latency_cycles = 300;
    cpu.memory_hierarchy.push_back(main_mem);
  }

  static size_t detectCacheSize(const std::string& cache_name) {
#ifdef __linux__
    // Try sysfs first
    for (int cpu = 0; cpu < 16; ++cpu) { // Check first 16 CPUs
      std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + 
                        "/cache/index";
      
      for (int idx = 0; idx < 4; ++idx) {
        std::string cache_path = path + std::to_string(idx);
        std::ifstream type_file(cache_path + "/type");
        std::ifstream level_file(cache_path + "/level");
        std::ifstream size_file(cache_path + "/size");
        
        std::string type, level, size_str;
        if (type_file >> type && level_file >> level && size_file >> size_str) {
          if ((cache_name == "L1d" && level == "1" && type == "Data") ||
              (cache_name == "L2" && level == "2") ||
              (cache_name == "L3" && level == "3")) {
            
            // Parse size (e.g., "32K", "256K", "8192K")
            size_t size = 0;
            if (size_str.back() == 'K') {
              size = std::stoi(size_str.substr(0, size_str.length()-1)) * 1024;
            } else if (size_str.back() == 'M') {
              size = std::stoi(size_str.substr(0, size_str.length()-1)) * 1024 * 1024;
            }
            
            if (size > 0) return size;
          }
        }
      }
    }
#endif
    return 0; // Not detected
  }

  static size_t getTotalMemory() {
#ifdef __linux__
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
      return info.totalram * info.mem_unit;
    }
#endif
    return 8ULL * 1024 * 1024 * 1024; // Default 8GB
  }

  static size_t estimateMemoryBandwidth() {
    // Very rough estimation based on typical values
    return 50; // GB/s - conservative estimate
  }

  static double estimateComputeThroughput(const TargetCharacteristics& cpu) {
    // Rough estimation: cores * frequency (assumed 2.5 GHz) * operations per cycle
    double ops_per_cycle = cpu.supports_vectorization ? 8.0 : 2.0; // Vector vs scalar
    return cpu.compute_units * 2.5 * ops_per_cycle; // GFLOPS
  }
};

/// GPU detection using various methods
class GPUDetector {
public:
  static std::vector<TargetCharacteristics> detectGPUs() {
    std::vector<TargetCharacteristics> gpus;
    
    // Try NVIDIA first
    auto nvidia_gpus = detectNVIDIAGPUs();
    gpus.insert(gpus.end(), nvidia_gpus.begin(), nvidia_gpus.end());
    
    // Try AMD
    auto amd_gpus = detectAMDGPUs();
    gpus.insert(gpus.end(), amd_gpus.begin(), amd_gpus.end());
    
    // Try Intel
    auto intel_gpus = detectIntelGPUs();
    gpus.insert(gpus.end(), intel_gpus.begin(), intel_gpus.end());
    
    return gpus;
  }

private:
  static std::vector<TargetCharacteristics> detectNVIDIAGPUs() {
    std::vector<TargetCharacteristics> gpus;
    
    // Try to detect via nvidia-smi or device files
#ifdef __linux__
    std::ifstream nvidia_dev("/proc/driver/nvidia/version");
    if (nvidia_dev.is_open()) {
      // NVIDIA driver is present, create a generic NVIDIA GPU profile
      TargetCharacteristics gpu;
      gpu.type = TargetType::GPU;
      gpu.name = "NVIDIA GPU";
      gpu.vendor = "NVIDIA";
      
      // Generic NVIDIA GPU characteristics
      gpu.compute_units = 2048; // Simplified SM count * cores per SM
      gpu.max_work_group_size = 1024;
      gpu.max_work_item_dimensions = 3;
      gpu.max_work_item_sizes = {1024, 1024, 64};
      
      // Memory hierarchy
      TargetCharacteristics::MemoryInfo shared_mem;
      shared_mem.level = MemoryLevel::SHARED;
      shared_mem.size_bytes = 48 * 1024; // 48KB shared memory
      shared_mem.bandwidth_gb_per_s = 8000;
      shared_mem.latency_cycles = 1;
      gpu.memory_hierarchy.push_back(shared_mem);
      
      TargetCharacteristics::MemoryInfo global_mem;
      global_mem.level = MemoryLevel::GLOBAL;
      global_mem.size_bytes = 8ULL * 1024 * 1024 * 1024; // 8GB VRAM
      global_mem.bandwidth_gb_per_s = 900; // High-end GPU bandwidth
      global_mem.latency_cycles = 400;
      gpu.memory_hierarchy.push_back(global_mem);
      
      gpu.supports_double_precision = true;
      gpu.supports_atomic_operations = true;
      gpu.supports_vectorization = true;
      gpu.supports_local_memory = true;
      
      gpu.peak_compute_throughput = 15000; // 15 TFLOPS
      gpu.memory_coalescing_factor = 0.95;
      
      // GPU-specific scheduling parameters
      gpu.scheduling_parameters["warp_size"] = 32;
      gpu.scheduling_parameters["max_threads_per_block"] = 1024;
      gpu.scheduling_parameters["preferred_tile_size"] = 16;
      gpu.scheduling_parameters["shared_memory_size"] = 49152;
      
      gpus.push_back(gpu);
    }
#endif
    
    return gpus;
  }

  static std::vector<TargetCharacteristics> detectAMDGPUs() {
    std::vector<TargetCharacteristics> gpus;
    
    // Try to detect AMD GPUs
#ifdef __linux__
    std::ifstream amd_dev("/sys/class/drm/card0/device/vendor");
    std::string vendor_id;
    if (amd_dev >> vendor_id && vendor_id == "0x1002") { // AMD vendor ID
      TargetCharacteristics gpu;
      gpu.type = TargetType::GPU;
      gpu.name = "AMD GPU";
      gpu.vendor = "AMD";
      
      // Generic AMD GPU characteristics
      gpu.compute_units = 2560; // CUs * cores per CU
      gpu.max_work_group_size = 256; // AMD typically uses smaller work groups
      gpu.max_work_item_dimensions = 3;
      gpu.max_work_item_sizes = {256, 256, 256};
      
      // Memory hierarchy
      TargetCharacteristics::MemoryInfo local_mem;
      local_mem.level = MemoryLevel::LOCAL;
      local_mem.size_bytes = 64 * 1024; // 64KB LDS
      local_mem.bandwidth_gb_per_s = 4000;
      local_mem.latency_cycles = 1;
      gpu.memory_hierarchy.push_back(local_mem);
      
      TargetCharacteristics::MemoryInfo global_mem;
      global_mem.level = MemoryLevel::GLOBAL;
      global_mem.size_bytes = 8ULL * 1024 * 1024 * 1024; // 8GB VRAM
      global_mem.bandwidth_gb_per_s = 512; // HBM bandwidth
      global_mem.latency_cycles = 300;
      gpu.memory_hierarchy.push_back(global_mem);
      
      gpu.supports_double_precision = true;
      gpu.supports_atomic_operations = true;
      gpu.supports_vectorization = true;
      gpu.supports_local_memory = true;
      
      gpu.peak_compute_throughput = 12000; // 12 TFLOPS
      gpu.memory_coalescing_factor = 0.90;
      
      gpu.scheduling_parameters["wavefront_size"] = 64;
      gpu.scheduling_parameters["preferred_tile_size"] = 16;
      gpu.scheduling_parameters["local_memory_size"] = 65536;
      
      gpus.push_back(gpu);
    }
#endif
    
    return gpus;
  }

  static std::vector<TargetCharacteristics> detectIntelGPUs() {
    // Intel GPU detection would go here
    return {};
  }
};

/// Enhanced target detector using all detection methods
class EnhancedTargetDetector : public TargetDetector {
public:
  std::vector<TargetCharacteristics> detectTargets() override {
    LLVM_DEBUG(llvm::dbgs() << "Starting enhanced target detection\n");
    
    std::vector<TargetCharacteristics> targets;
    
    // Always detect CPU with advanced detection
    if (SUPPORT_CPU) {
      targets.push_back(AdvancedCPUDetector::detectCPU());
      LLVM_DEBUG(llvm::dbgs() << "Detected CPU target\n");
    }
    
    // Detect GPUs
    if (SUPPORT_GPU) {
      auto gpus = GPUDetector::detectGPUs();
      targets.insert(targets.end(), gpus.begin(), gpus.end());
      LLVM_DEBUG(llvm::dbgs() << "Detected " << gpus.size() << " GPU targets\n");
    }
    
    // TODO: Add OpenCL, FPGA, and other accelerator detection
    
    return targets;
  }

  TargetCharacteristics getDefaultTarget() override {
    auto targets = detectTargets();
    
    if (targets.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No targets detected, using fallback\n");
      return createFallbackTarget();
    }
    
    // Priority: GPU > CPU > others
    for (const auto& target : targets) {
      if (target.type == TargetType::GPU) {
        LLVM_DEBUG(llvm::dbgs() << "Selected GPU as default target: " << target.name << "\n");
        return target;
      }
    }
    
    for (const auto& target : targets) {
      if (target.type == TargetType::CPU) {
        LLVM_DEBUG(llvm::dbgs() << "Selected CPU as default target: " << target.name << "\n");
        return target;
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Selected first available target: " << targets[0].name << "\n");
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
    
    LLVM_DEBUG(llvm::dbgs() << "Target '" << name << "' not found, using default\n");
    return getDefaultTarget();
  }

private:
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

// Override the factory to use enhanced detection
std::unique_ptr<TargetDetector> TargetDetectorFactory::createDetector() {
  return std::make_unique<EnhancedTargetDetector>();
}

} // namespace target
} // namespace autopoly
