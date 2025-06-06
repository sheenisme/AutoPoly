//===- DependenceAnalysis.h - Polyhedral Dependence Analysis --------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file defines the dependence analysis framework for polyhedral models,
// including data dependences, memory dependences, and other constraints
// that affect scheduling transformations.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOPOLY_ANALYSIS_DEPENDENCEANALYSIS_H
#define AUTOPOLY_ANALYSIS_DEPENDENCEANALYSIS_H

#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "mlir/Support/LLVM.h"

#include <memory>
#include <vector>
#include <set>
#include <map>

// Forward declarations for ISL structures
struct isl_ctx;
struct isl_union_map;
struct isl_map;
struct isl_set;

namespace autopoly {
namespace analysis {

/// Types of dependence relationships
enum class DependenceType {
  RAW,        ///< Read-After-Write (true dependence)
  WAR,        ///< Write-After-Read (anti dependence)
  WAW,        ///< Write-After-Write (output dependence)
  CONTROL,    ///< Control dependence
  MEMORY,     ///< Memory dependence (aliasing)
  REDUCTION   ///< Reduction dependence
};

/// Distance vector for loop-carried dependences
struct DependenceVector {
  std::vector<int> distances;          ///< Distance per loop level
  std::vector<bool> is_distance_known; ///< Whether distance is precisely known
  DependenceType type;                 ///< Type of dependence
  
  /// Check if dependence is loop-independent
  bool isLoopIndependent() const;
  
  /// Check if dependence is lexicographically positive
  bool isLexicographicallyPositive() const;
  
  /// Get the outermost loop level with non-zero distance
  int getCarryingLoop() const;
};

/// Individual dependence relation between two statements
struct DependenceRelation {
  std::string source_statement;        ///< Source statement name
  std::string target_statement;        ///< Target statement name
  DependenceType type;                 ///< Type of dependence
  isl_map* relation_map;               ///< ISL map representing the dependence
  mlir::Value shared_array;            ///< Array involved in the dependence
  
  /// Dependence analysis results
  std::vector<DependenceVector> distance_vectors;
  bool is_uniform = false;             ///< Uniform dependence distance
  bool is_scalable = true;             ///< Scales with problem size
  
  /// Scheduling constraints
  bool prevents_parallelization = false;  ///< Blocks parallel execution
  std::vector<int> blocked_dimensions;    ///< Dimensions blocked by this dependence
  
  /// Transformation properties
  bool can_be_carried_by_tiling = false; ///< Can be satisfied by tiling
  bool can_be_privatized = false;        ///< Array can be privatized
  int min_skewing_factor = 0;            ///< Minimum skewing to satisfy
};

/// Complete dependence information for a polyhedral model
class DependenceInfo {
public:
  /// Constructor
  explicit DependenceInfo(isl_ctx* ctx);
  
  /// Destructor
  ~DependenceInfo();
  
  /// Get all dependence relations
  const std::vector<DependenceRelation>& getDependences() const { return dependences_; }
  
  /// Get dependences of a specific type
  std::vector<DependenceRelation> getDependencesByType(DependenceType type) const;
  
  /// Get dependences involving a specific statement
  std::vector<DependenceRelation> getDependencesForStatement(
      const std::string& statement) const;
  
  /// Get dependences involving a specific array
  std::vector<DependenceRelation> getDependencesForArray(mlir::Value array) const;
  
  /// Add a dependence relation
  void addDependence(const DependenceRelation& dep);
  
  /// Check if two statements have any dependence
  bool haveDependence(const std::string& source, const std::string& target) const;
  
  /// Get the combined dependence map for scheduling
  isl_union_map* getCombinedDependenceMap() const;
  
  /// Check if a loop dimension can be parallelized
  bool canParallelizeDimension(int dimension) const;
  
  /// Get the minimum tiling factor required for correctness
  std::vector<int> getMinimumTilingSizes() const;
  
  /// 
  bool canFuse(const std::string& stmtA, const std::string& stmtB) const;
  
  /// 
  bool hasLoopCarriedDependences() const;

private:
  isl_ctx* ctx_;
  std::vector<DependenceRelation> dependences_;
  
  /// Cached analysis results
  mutable isl_union_map* combined_dependence_map_;
  mutable bool analysis_cache_valid_;
  
  /// Helper methods
  void invalidateCache();
  void computeCombinedDependenceMap() const;
};

/// Main dependence analyzer class
class DependenceAnalyzer {
public:
  /// Constructor with ISL context
  explicit DependenceAnalyzer(isl_ctx* ctx);
  
  /// Destructor
  ~DependenceAnalyzer();
  
  /// Perform complete dependence analysis on a polyhedral model
  std::unique_ptr<DependenceInfo> analyze(const PolyhedralModel& model);
  
  /// Analyze dependences between specific statements
  std::vector<DependenceRelation> analyzeStatementPair(
      const PolyhedralStatement& source,
      const PolyhedralStatement& target,
      const PolyhedralModel& model);
  
  /// Analyze array-based dependences
  std::vector<DependenceRelation> analyzeArrayDependences(
      const ArrayAccessInfo& array_info,
      const PolyhedralModel& model);
  
  /// Compute distance vectors for a dependence
  std::vector<DependenceVector> computeDistanceVectors(
      const DependenceRelation& dependence);

  /// Create a simple dependence map between two statements
  isl_map* createSimpleDependenceMap(const PolyhedralStatement& source,
                                                      const PolyhedralStatement& target);
private:
  isl_ctx* ctx_;
  
  /// Analysis configuration
  bool enable_may_dependences_ = true;    ///< Include may-dependences
  bool enable_scalar_dependences_ = true; ///< Include scalar dependences
  bool precise_analysis_ = true;          ///< Use precise analysis methods
  
  /// Helper methods for different dependence types
  std::vector<DependenceRelation> computeRAWDependences(const PolyhedralModel& model);
  std::vector<DependenceRelation> computeWARDependences(const PolyhedralModel& model);
  std::vector<DependenceRelation> computeWAWDependences(const PolyhedralModel& model);
  std::vector<DependenceRelation> computeControlDependences(const PolyhedralModel& model);
  std::vector<DependenceRelation> computeReductionDependences(const PolyhedralModel& model);
  
  /// ISL-based analysis methods
  isl_union_map* computeAccessMap(const PolyhedralModel& model, bool reads_only);
  isl_union_map* computeScheduleMap(const PolyhedralModel& model);
  
  /// Dependence classification and analysis
  DependenceType classifyDependence(isl_map* dep_map,
                                   const PolyhedralStatement& source,
                                   const PolyhedralStatement& target);
  
  bool isUniformDependence(isl_map* dep_map);
  bool canBePrivatized(const DependenceRelation& dep, const PolyhedralModel& model);
  bool canBeCarriedByTiling(const DependenceRelation& dep);
  
  /// Distance vector computation
  std::vector<int> extractDistanceVector(isl_map* dep_map);
  bool isDistanceKnown(isl_map* dep_map, int dimension);
};

/// Utility functions for dependence analysis
class DependenceUtils {
public:
  /// Convert dependence type to string
  static std::string dependenceTypeToString(DependenceType type);
  
  /// Check if a dependence type allows parallelization
  static bool allowsParallelization(DependenceType type);
  
  /// Compute the minimum distance for a set of dependences
  static std::vector<int> computeMinimumDistances(
      const std::vector<DependenceRelation>& dependences);
  
  /// Check if dependences form a cycle
  static bool formsCycle(const std::vector<DependenceRelation>& dependences);
  
  /// Compute strongly connected components of dependences
  static std::vector<std::set<std::string>> computeSCC(
      const std::vector<DependenceRelation>& dependences);
  
  /// Estimate memory traffic reduction from tiling
  static double estimateMemoryTrafficReduction(
      const DependenceInfo& deps,
      const std::vector<int>& tile_sizes);
  
  /// Find parallelizable loop levels
  static std::vector<int> findParallelizableLoops(const DependenceInfo& deps);
  
  /// Compute optimal privatization candidates
  static std::vector<mlir::Value> computePrivatizationCandidates(
      const DependenceInfo& deps,
      const PolyhedralModel& model);
};

/// Dependence-aware transformation validator
class TransformationValidator {
public:
  /// Constructor
  explicit TransformationValidator(const DependenceInfo& deps);
  
  /// Validate that a tiling transformation preserves dependences
  bool validateTiling(const std::vector<int>& tile_sizes) const;
  
  /// Validate that loop interchange preserves dependences
  bool validateInterchange(const std::vector<int>& new_order) const;
  
  /// Validate that skewing transformation preserves dependences
  bool validateSkewing(const std::vector<std::vector<int>>& skewing_matrix) const;
  
  /// Validate that parallelization is safe for given dimensions
  bool validateParallelization(const std::vector<int>& parallel_dims) const;
  
  /// Validate fusion of loop nests
  bool validateFusion(const std::vector<std::string>& statements_to_fuse) const;

private:
  const DependenceInfo& deps_;
  
  /// Helper validation methods
  bool checkDependencePreservation(isl_union_map* transformation) const;
  bool checkLexicographicOrder(isl_union_map* transformation) const;
};

} // namespace analysis
} // namespace autopoly

#endif // AUTOPOLY_ANALYSIS_DEPENDENCEANALYSIS_H
