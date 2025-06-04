//===- SchedulingStrategy.h - Scheduling Strategy Framework -*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file defines the scheduling strategy framework that maps target
// hardware characteristics to appropriate scheduling algorithms and
// optimization parameters.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOPOLY_SCHEDULING_SCHEDULINGSTRATEGY_H
#define AUTOPOLY_SCHEDULING_SCHEDULINGSTRATEGY_H

#include "AutoPoly/Target/TargetInfo.h"
#include <memory>
#include <string>
#include <vector>
#include <map>

// Forward declarations for ISL and PPCG
struct isl_ctx;
struct isl_schedule;
struct isl_schedule_node;
struct ppcg_options;

namespace autopoly {
namespace scheduling {

/// Enumeration of available scheduling algorithms
enum class SchedulingAlgorithm {
  ISL_SCHEDULER,        ///< ISL's built-in scheduler
  FEAUTRIER,           ///< Feautrier's scheduling algorithm
  PLUTO,               ///< PLUTO algorithm
  PPCG_DEFAULT,        ///< PPCG's default scheduling
  CUSTOM               ///< Custom scheduling algorithm
};

/// Optimization techniques that can be applied
enum class OptimizationTechnique {
  TILING,              ///< Loop tiling transformation
  FUSION,              ///< Loop fusion
  SKEWING,             ///< Loop skewing
  PARALLELIZATION,     ///< Parallel loop generation
  VECTORIZATION,       ///< Vector/SIMD optimization
  UNROLLING,           ///< Loop unrolling
  INTERCHANGE,         ///< Loop interchange
  STRIP_MINING,        ///< Strip mining
  PRIVATIZATION        ///< Array privatization
};

/// Parameters for scheduling transformations
struct SchedulingParameters {
  /// Tiling parameters
  std::vector<int> tile_sizes;
  bool enable_tiling = true;
  
  /// Parallelization parameters
  std::vector<int> parallel_dimensions;
  int max_parallel_depth = 3;
  bool enable_nested_parallelism = false;
  
  /// Fusion parameters
  bool enable_loop_fusion = true;
  int fusion_strategy = 0; // 0: greedy, 1: maximal, 2: minimal
  
  /// Unrolling parameters
  std::vector<int> unroll_factors;
  bool enable_unrolling = true;
  
  /// Skewing parameters
  bool enable_skewing = false;
  std::vector<std::vector<int>> skewing_factors;
  
  /// Memory optimization parameters
  bool enable_array_privatization = true;
  bool enable_copy_optimization = true;
  
  /// Target-specific parameters
  std::map<std::string, int> target_specific_params;
};

/// Abstract base class for scheduling strategies
class SchedulingStrategy {
public:
  virtual ~SchedulingStrategy() = default;
  
  /// Get the name of this scheduling strategy
  virtual std::string getName() const = 0;
  
  /// Check if this strategy is suitable for the given target
  virtual bool isSuitableForTarget(const target::TargetCharacteristics& target) const = 0;
  
  /// Get recommended scheduling parameters for the target
  virtual SchedulingParameters getParameters(
      const target::TargetCharacteristics& target) const = 0;
  
  /// Get the primary scheduling algorithm to use
  virtual SchedulingAlgorithm getPrimaryAlgorithm() const = 0;
  
  /// Get the optimization techniques to apply
  virtual std::vector<OptimizationTechnique> getOptimizationTechniques() const = 0;
  
  /// Calculate priority score for target selection (higher = better)
  virtual double calculatePriority(const target::TargetCharacteristics& target) const = 0;
};

/// CPU-specific scheduling strategy
class CPUSchedulingStrategy : public SchedulingStrategy {
public:
  std::string getName() const override { return "CPU"; }
  bool isSuitableForTarget(const target::TargetCharacteristics& target) const override;
  SchedulingParameters getParameters(const target::TargetCharacteristics& target) const override;
  SchedulingAlgorithm getPrimaryAlgorithm() const override;
  std::vector<OptimizationTechnique> getOptimizationTechniques() const override;
  double calculatePriority(const target::TargetCharacteristics& target) const override;
};

/// GPU-specific scheduling strategy
class GPUSchedulingStrategy : public SchedulingStrategy {
public:
  std::string getName() const override { return "GPU"; }
  bool isSuitableForTarget(const target::TargetCharacteristics& target) const override;
  SchedulingParameters getParameters(const target::TargetCharacteristics& target) const override;
  SchedulingAlgorithm getPrimaryAlgorithm() const override;
  std::vector<OptimizationTechnique> getOptimizationTechniques() const override;
  double calculatePriority(const target::TargetCharacteristics& target) const override;
};

/// OpenCL-specific scheduling strategy
class OpenCLSchedulingStrategy : public SchedulingStrategy {
public:
  std::string getName() const override { return "OpenCL"; }
  bool isSuitableForTarget(const target::TargetCharacteristics& target) const override;
  SchedulingParameters getParameters(const target::TargetCharacteristics& target) const override;
  SchedulingAlgorithm getPrimaryAlgorithm() const override;
  std::vector<OptimizationTechnique> getOptimizationTechniques() const override;
  double calculatePriority(const target::TargetCharacteristics& target) const override;
};

/// FPGA-specific scheduling strategy
class FPGASchedulingStrategy : public SchedulingStrategy {
public:
  std::string getName() const override { return "FPGA"; }
  bool isSuitableForTarget(const target::TargetCharacteristics& target) const override;
  SchedulingParameters getParameters(const target::TargetCharacteristics& target) const override;
  SchedulingAlgorithm getPrimaryAlgorithm() const override;
  std::vector<OptimizationTechnique> getOptimizationTechniques() const override;
  double calculatePriority(const target::TargetCharacteristics& target) const override;
};

/// Manager for scheduling strategies - implements strategy selection logic
class SchedulingStrategyManager {
public:
  /// Constructor - initializes all available strategies
  SchedulingStrategyManager();
  
  /// Destructor
  ~SchedulingStrategyManager();
  
  /// Register a custom scheduling strategy
  void registerStrategy(std::unique_ptr<SchedulingStrategy> strategy);
  
  /// Select the best strategy for a given target
  const SchedulingStrategy* selectStrategy(const target::TargetCharacteristics& target) const;
  
  /// Get all available strategies
  std::vector<const SchedulingStrategy*> getAllStrategies() const;
  
  /// Get strategy by name
  const SchedulingStrategy* getStrategyByName(const std::string& name) const;
  
  /// Force use of a specific strategy (for testing/debugging)
  void forceStrategy(const std::string& name);
  
  /// Clear forced strategy selection
  void clearForcedStrategy();

private:
  std::vector<std::unique_ptr<SchedulingStrategy>> strategies_;
  std::string forced_strategy_name_;
  
  /// Initialize built-in strategies
  void initializeBuiltinStrategies();
};

/// Utility functions for scheduling strategies
class SchedulingUtils {
public:
  /// Convert scheduling algorithm enum to string
  static std::string algorithmToString(SchedulingAlgorithm algorithm);
  
  /// Convert optimization technique enum to string
  static std::string techniqueToString(OptimizationTechnique technique);
  
  /// Validate scheduling parameters for a target
  static bool validateParameters(const SchedulingParameters& params,
                                const target::TargetCharacteristics& target);
  
  /// Merge scheduling parameters (second overrides first)
  static SchedulingParameters mergeParameters(const SchedulingParameters& base,
                                            const SchedulingParameters& override);
  
  /// Calculate optimal tile sizes based on memory hierarchy
  static std::vector<int> calculateOptimalTileSizes(
      const target::TargetCharacteristics& target,
      const std::vector<int>& loop_bounds,
      int data_element_size);
};

} // namespace scheduling
} // namespace autopoly

#endif // AUTOPOLY_SCHEDULING_SCHEDULINGSTRATEGY_H
