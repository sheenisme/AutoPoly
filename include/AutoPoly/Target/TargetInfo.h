//===- TargetInfo.h - Target Information Definition ------------------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file defines target information for the AutoPoly framework.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOPOLY_TARGET_TARGETINFO_H
#define AUTOPOLY_TARGET_TARGETINFO_H

#include <memory>
#include <string>
#include <vector>
#include <map>

namespace autopoly {
namespace target {

/// Enumeration of supported target hardware types
enum class TargetType {
  CPU,        ///< General-purpose CPU
  GPU,        ///< Graphics Processing Unit
  OPENCL,     ///< OpenCL-compatible device
  FPGA,       ///< Field-Programmable Gate Array
  CGRA,       ///< Coarse-Grained Reconfigurable Array
  NPU,        ///< Neural Processing Unit
  DPU,        ///< Deep Processing Unit
  PIM,        ///< Processing-in-Memory
  UNKNOWN     ///< Unknown or unsupported target
};

/// Memory hierarchy level enumeration
enum class MemoryLevel {
  GLOBAL,     ///< Global memory (off-chip)
  SHARED,     ///< Shared memory (on-chip, shared across cores)
  LOCAL,      ///< Local memory (per-core)
  REGISTER    ///< Register memory (fastest)
};

/// Hardware characteristics for a specific target
struct TargetCharacteristics {
  TargetType type;
  std::string name;
  std::string vendor;
  
  /// Compute capabilities
  int compute_units;              ///< Number of compute units/cores
  int max_work_group_size;        ///< Maximum work group size
  int max_work_item_dimensions;   ///< Maximum work item dimensions
  std::vector<int> max_work_item_sizes; ///< Max work item sizes per dimension
  
  /// Memory hierarchy information
  struct MemoryInfo {
    MemoryLevel level;
    size_t size_bytes;            ///< Memory size in bytes
    size_t bandwidth_gb_per_s;    ///< Memory bandwidth in GB/s
    int latency_cycles;           ///< Access latency in cycles
    
    bool operator==(const MemoryInfo& other) const {
      return level == other.level &&
             size_bytes == other.size_bytes &&
             bandwidth_gb_per_s == other.bandwidth_gb_per_s &&
             latency_cycles == other.latency_cycles;
    }
  };
  std::vector<MemoryInfo> memory_hierarchy;
  
  /// Architecture-specific features
  bool supports_double_precision;  ///< Double precision floating point
  bool supports_atomic_operations; ///< Atomic memory operations
  bool supports_vectorization;     ///< Vector/SIMD operations
  bool supports_local_memory;      ///< Local/scratch memory
  
  /// Performance characteristics
  double peak_compute_throughput;  ///< Peak compute throughput (GFLOPS)
  double memory_coalescing_factor; ///< Memory coalescing efficiency
  
  /// Target-specific parameters for scheduling
  std::map<std::string, int> scheduling_parameters;
};

/// Base class for target detection and characterization
class TargetDetector {
public:
  virtual ~TargetDetector() = default;
  
  /// Detect available targets in the system
  virtual std::vector<TargetCharacteristics> detectTargets() = 0;
  
  /// Get default target (highest priority available target)
  virtual TargetCharacteristics getDefaultTarget() = 0;
  
  /// Check if a specific target type is available
  virtual bool isTargetAvailable(TargetType type) = 0;
  
  /// Get target characteristics by name
  virtual TargetCharacteristics getTargetByName(const std::string& name) = 0;
};

/// Factory for creating target detectors
class TargetDetectorFactory {
public:
  /// Create a target detector for the current system
  static std::unique_ptr<TargetDetector> createDetector();
  
  /// Create a mock target detector for testing
  static std::unique_ptr<TargetDetector> createMockDetector(
      const std::vector<TargetCharacteristics>& targets);
};

/// Utility functions for target information
class TargetUtils {
public:
  /// Convert target type to string
  static std::string targetTypeToString(TargetType type);
  
  /// Convert string to target type
  static TargetType stringToTargetType(const std::string& str);
  
  /// Get recommended tile sizes for a target
  static std::vector<int> getRecommendedTileSizes(
      const TargetCharacteristics& target, int num_dimensions);
  
  /// Get recommended unroll factors for a target
  static std::vector<int> getRecommendedUnrollFactors(
      const TargetCharacteristics& target, int num_loops);
  
  /// Calculate memory hierarchy optimization parameters
  static std::map<MemoryLevel, int> calculateMemoryParameters(
      const TargetCharacteristics& target);
};

} // namespace target
} // namespace autopoly

#endif // AUTOPOLY_TARGET_TARGETINFO_H
