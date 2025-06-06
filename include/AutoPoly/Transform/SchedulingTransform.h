//===- SchedulingTransform.h - Polyhedral Scheduling Transform ------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file defines the scheduling transformation framework that applies
// polyhedral transformations based on target characteristics and
// dependence constraints.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOPOLY_TRANSFORM_SCHEDULINGTRANSFORM_H
#define AUTOPOLY_TRANSFORM_SCHEDULINGTRANSFORM_H

#include "AutoPoly/Analysis/DependenceAnalysis.h"
#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "AutoPoly/Scheduling/SchedulingStrategy.h"
#include "AutoPoly/Target/TargetInfo.h"

#include <memory>
#include <vector>
#include <map>

// Forward declarations for ISL and PPCG
struct isl_ctx;
struct isl_schedule;
struct isl_schedule_node;
struct isl_union_map;
struct ppcg_options;
struct ppcg_scop;

namespace autopoly {
namespace transform {

/// Result of a scheduling transformation
struct TransformationResult {
  isl_schedule* transformed_schedule;     ///< New schedule after transformation
  bool transformation_successful = false; ///< Whether transformation succeeded
  std::string error_message;              ///< Error message if failed
  
  /// Transformation metrics
  double estimated_speedup = 1.0;        ///< Estimated performance improvement
  int parallelism_degree = 1;            ///< Degree of parallelism achieved
  std::vector<int> final_tile_sizes;     ///< Applied tile sizes
  std::vector<int> parallel_dimensions;  ///< Parallelized dimensions
  
  /// Memory optimization results
  double memory_traffic_reduction = 0.0; ///< Estimated memory traffic reduction
  bool array_privatization_applied = false; ///< Whether privatization was used
  std::vector<std::string> privatized_arrays; ///< Names of privatized arrays
  
  /// Applied transformations
  std::vector<scheduling::OptimizationTechnique> applied_techniques;
  std::map<std::string, int> transformation_parameters;
};

/// Base class for individual scheduling transformations
class SchedulingTransformation {
public:
  virtual ~SchedulingTransformation() = default;
  
  /// Get the name of this transformation
  virtual std::string getName() const = 0;
  
  /// Check if this transformation is applicable
  virtual bool isApplicable(const analysis::PolyhedralModel& model,
                           const analysis::DependenceInfo& deps,
                           const target::TargetCharacteristics& target) const = 0;
  
  /// Apply the transformation
  virtual TransformationResult apply(isl_schedule* schedule,
                                   const analysis::PolyhedralModel& model,
                                   const analysis::DependenceInfo& deps,
                                   const scheduling::SchedulingParameters& params) const = 0;
  
  /// Estimate the benefit of applying this transformation
  virtual double estimateBenefit(const analysis::PolyhedralModel& model,
                               const analysis::DependenceInfo& deps,
                               const target::TargetCharacteristics& target) const = 0;
};

/// Tiling transformation
class TilingTransformation : public SchedulingTransformation {
public:
  std::string getName() const override { return "Tiling"; }
  
  bool isApplicable(const analysis::PolyhedralModel& model,
                   const analysis::DependenceInfo& deps,
                   const target::TargetCharacteristics& target) const override;
  
  TransformationResult apply(isl_schedule* schedule,
                           const analysis::PolyhedralModel& model,
                           const analysis::DependenceInfo& deps,
                           const scheduling::SchedulingParameters& params) const override;
  
  double estimateBenefit(const analysis::PolyhedralModel& model,
                       const analysis::DependenceInfo& deps,
                       const target::TargetCharacteristics& target) const override;

private:
  /// Helper methods for tiling
  std::vector<int> computeOptimalTileSizes(const analysis::PolyhedralModel& model,
                                         const target::TargetCharacteristics& target) const;
  
  isl_schedule* applyTiling(isl_schedule* schedule,
                          const std::vector<int>& tile_sizes) const;
  
  bool validateTilingSizes(const std::vector<int>& tile_sizes,
                         const analysis::DependenceInfo& deps) const;
};

/// Parallelization transformation
class ParallelizationTransformation : public SchedulingTransformation {
public:
  std::string getName() const override { return "Parallelization"; }
  
  bool isApplicable(const analysis::PolyhedralModel& model,
                   const analysis::DependenceInfo& deps,
                   const target::TargetCharacteristics& target) const override;
  
  TransformationResult apply(isl_schedule* schedule,
                           const analysis::PolyhedralModel& model,
                           const analysis::DependenceInfo& deps,
                           const scheduling::SchedulingParameters& params) const override;
  
  double estimateBenefit(const analysis::PolyhedralModel& model,
                       const analysis::DependenceInfo& deps,
                       const target::TargetCharacteristics& target) const override;

private:
  /// Find parallelizable dimensions
  std::vector<int> findParallelizableDimensions(const analysis::DependenceInfo& deps) const;
  
  /// Apply parallel annotation to schedule
  isl_schedule* markParallelDimensions(isl_schedule* schedule,
                                     const std::vector<int>& parallel_dims) const;
  
  /// Estimate parallel efficiency
  double estimateParallelEfficiency(const std::vector<int>& parallel_dims,
                                  const target::TargetCharacteristics& target) const;
};

/// Loop fusion transformation
class FusionTransformation : public SchedulingTransformation {
public:
  std::string getName() const override { return "Fusion"; }
  
  bool isApplicable(const analysis::PolyhedralModel& model,
                   const analysis::DependenceInfo& deps,
                   const target::TargetCharacteristics& target) const override;
  
  TransformationResult apply(isl_schedule* schedule,
                           const analysis::PolyhedralModel& model,
                           const analysis::DependenceInfo& deps,
                           const scheduling::SchedulingParameters& params) const override;
  
  double estimateBenefit(const analysis::PolyhedralModel& model,
                       const analysis::DependenceInfo& deps,
                       const target::TargetCharacteristics& target) const override;

private:
  /// Find fusable statement groups
  std::vector<std::vector<std::string>> findFusableGroups(
      const analysis::PolyhedralModel& model,
      const analysis::DependenceInfo& deps) const;
  
  /// Apply fusion to schedule
  isl_schedule* applyFusion(isl_schedule* schedule,
                          const std::vector<std::string>& statements) const;
};

/// Skewing transformation
class SkewingTransformation : public SchedulingTransformation {
public:
  std::string getName() const override { return "Skewing"; }
  
  bool isApplicable(const analysis::PolyhedralModel& model,
                   const analysis::DependenceInfo& deps,
                   const target::TargetCharacteristics& target) const override;
  
  TransformationResult apply(isl_schedule* schedule,
                           const analysis::PolyhedralModel& model,
                           const analysis::DependenceInfo& deps,
                           const scheduling::SchedulingParameters& params) const override;
  
  double estimateBenefit(const analysis::PolyhedralModel& model,
                       const analysis::DependenceInfo& deps,
                       const target::TargetCharacteristics& target) const override;

private:
  /// Compute skewing factors
  std::vector<std::vector<int>> computeSkewingFactors(
      const analysis::DependenceInfo& deps) const;
  
  /// Apply skewing transformation
  isl_schedule* applySkewing(isl_schedule* schedule,
                           const std::vector<std::vector<int>>& skewing_matrix) const;
};

/// Main scheduling transformer that coordinates all transformations
class SchedulingTransformer {
public:
  /// Constructor
  explicit SchedulingTransformer(isl_ctx* ctx);
  
  /// Destructor
  ~SchedulingTransformer();
  
  /// Perform complete scheduling transformation
  TransformationResult transform(const analysis::PolyhedralModel& model,
                               const analysis::DependenceInfo& deps,
                               const target::TargetCharacteristics& target,
                               const scheduling::SchedulingStrategy& strategy);
  
  /// Apply specific transformation sequence
  TransformationResult transformWithSequence(
      const analysis::PolyhedralModel& model,
      const analysis::DependenceInfo& deps,
      const std::vector<scheduling::OptimizationTechnique>& techniques,
      const scheduling::SchedulingParameters& params);
  
  /// Register a custom transformation
  void registerTransformation(std::unique_ptr<SchedulingTransformation> transform);
  
  /// Get available transformations
  std::vector<const SchedulingTransformation*> getAvailableTransformations() const;

private:
  isl_ctx* ctx_;
  std::vector<std::unique_ptr<SchedulingTransformation>> transformations_;
  
  /// PPCG integration
  ppcg_scop* createPPCGScop(const analysis::PolyhedralModel& model,
                           const analysis::DependenceInfo& deps);
  
  isl_schedule* applyPPCGScheduling(ppcg_scop* scop,
                                  const scheduling::SchedulingParameters& params,
                                  const target::TargetCharacteristics& target);
  
  /// Transformation sequence planning
  std::vector<const SchedulingTransformation*> planTransformationSequence(
      const analysis::PolyhedralModel& model,
      const analysis::DependenceInfo& deps,
      const target::TargetCharacteristics& target,
      const std::vector<scheduling::OptimizationTechnique>& techniques);
  
  /// Transformation application
  TransformationResult applyTransformationSequence(
      isl_schedule* initial_schedule,
      const std::vector<const SchedulingTransformation*>& sequence,
      const analysis::PolyhedralModel& model,
      const analysis::DependenceInfo& deps,
      const scheduling::SchedulingParameters& params);
  
  /// Performance estimation
  double estimateOverallSpeedup(const TransformationResult& result,
                              const target::TargetCharacteristics& target) const;
  
  /// Schedule validation
  bool validateTransformedSchedule(isl_schedule* schedule,
                                 const analysis::DependenceInfo& deps) const;
  
  /// Initialize built-in transformations
  void initializeBuiltinTransformations();
};

/// Integration with PPCG for GPU-specific optimizations
class PPCGIntegration {
public:
  /// Constructor with ISL context
  explicit PPCGIntegration(isl_ctx* ctx);
  
  /// Destructor
  ~PPCGIntegration();
  
  /// Generate GPU-optimized schedule using PPCG
  TransformationResult generateGPUSchedule(const analysis::PolyhedralModel& model,
                                         const analysis::DependenceInfo& deps,
                                         const target::TargetCharacteristics& target);
  
  /// Generate OpenCL-optimized schedule using PPCG
  TransformationResult generateOpenCLSchedule(const analysis::PolyhedralModel& model,
                                            const analysis::DependenceInfo& deps,
                                            const target::TargetCharacteristics& target);

private:
  isl_ctx* ctx_;
  ppcg_options* default_options_;
  
  /// Configure PPCG options for target
  ppcg_options* configurePPCGOptions(const target::TargetCharacteristics& target);
  
  /// Convert polyhedral model to PPCG format
  ppcg_scop* convertToPPCGScop(const analysis::PolyhedralModel& model,
                              const analysis::DependenceInfo& deps);
  
  /// Extract results from PPCG
  TransformationResult extractPPCGResults(ppcg_scop* scop,
                                        const target::TargetCharacteristics& target);
};

/// Utility functions for scheduling transformations
class TransformUtils {
public:
  /// Validate that a schedule preserves dependences
  static bool validateSchedule(isl_schedule* schedule,
                             const analysis::DependenceInfo& deps);
  
  /// Compute the parallelism degree of a schedule
  static int computeParallelismDegree(isl_schedule* schedule);
  
  /// Estimate memory access cost for a schedule
  static double estimateMemoryCost(isl_schedule* schedule,
                                 const analysis::PolyhedralModel& model,
                                 const target::TargetCharacteristics& target);
  
  /// Extract tile sizes from a tiled schedule
  static std::vector<int> extractTileSizes(isl_schedule* schedule);
  
  /// Check if schedule contains parallel dimensions
  static std::vector<int> extractParallelDimensions(isl_schedule* schedule);
  
  /// Merge transformation results
  static TransformationResult mergeResults(const std::vector<TransformationResult>& results);
  
  /// Convert ISL schedule to string representation
  static std::string scheduleToString(isl_schedule* schedule);
};

} // namespace transform
} // namespace autopoly

#endif // AUTOPOLY_TRANSFORM_SCHEDULINGTRANSFORM_H
