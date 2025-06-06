//===- PolyhedralExtraction.h - MLIR to Polyhedral Model ------------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file defines the extraction of polyhedral model information from
// MLIR affine dialect operations, creating ISL schedule trees for
// scheduling transformation.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOPOLY_ANALYSIS_POLYHEDRALEXTRACTION_H
#define AUTOPOLY_ANALYSIS_POLYHEDRALEXTRACTION_H

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <memory>
#include <vector>
#include <map>
#include <set>

// Forward declarations for ISL structures
struct isl_ctx;
struct isl_schedule;
struct isl_schedule_tree;
struct isl_schedule_node;
struct isl_union_map;
struct isl_union_set;
struct isl_set;
struct isl_map;

namespace mlir {
class MemRefType;
} // namespace mlir

namespace autopoly {
namespace analysis {

/// Information about a polyhedral statement (basic block)
struct PolyhedralStatement {
  mlir::Operation* operation;           ///< Original MLIR operation
  std::string name;                     ///< Unique statement name
  isl_set* domain;                      ///< Iteration domain
  std::vector<isl_map*> access_maps;    ///< Memory access patterns
  std::vector<mlir::Value> accessed_arrays; ///< Accessed memory references
  
  /// Statement properties
  bool is_read_only = false;            ///< Only reads from memory
  bool is_write_only = false;           ///< Only writes to memory
  bool has_side_effects = false;        ///< Has side effects beyond memory access
  
  /// Performance hints
  int computational_intensity = 1;      ///< Relative computational cost
  std::vector<int> vectorizable_dimensions; ///< Dimensions suitable for vectorization
  std::string getName() const { return name; }
};

/// Information about array accesses within the polyhedral model
struct ArrayAccessInfo {
  mlir::Value array;                    ///< Array being accessed
  mlir::MemRefType array_type;          ///< Type information
  std::vector<isl_map*> read_accesses;  ///< Read access patterns
  std::vector<isl_map*> write_accesses; ///< Write access patterns
  
  /// Access pattern analysis
  bool is_affine = true;                ///< All accesses are affine
  bool is_contiguous = false;           ///< Accesses are contiguous
  std::vector<int> stride_patterns;     ///< Stride patterns per dimension
  
  /// Memory reuse analysis
  bool has_temporal_reuse = false;      ///< Temporal locality present
  bool has_spatial_reuse = false;       ///< Spatial locality present
  int reuse_distance = -1;              ///< Average reuse distance
};

/// Extracted polyhedral model from MLIR affine operations
class PolyhedralModel {
public:
  /// Constructor with ISL context
  explicit PolyhedralModel(isl_ctx* ctx);
  
  /// Destructor - cleans up ISL objects
  ~PolyhedralModel();
  
  /// Get the ISL context
  isl_ctx* getContext() const { return ctx_; }
  
  /// Get the schedule tree
  isl_schedule* getSchedule() const { return schedule_; }
  
  /// Get all statements
  const std::vector<PolyhedralStatement>& getStatements() const { return statements_; }
  
  /// Get array access information
  const std::vector<ArrayAccessInfo>& getArrayAccesses() const { return array_accesses_; }
  
  /// Get iteration domain union
  isl_union_set* getIterationDomain() const { return iteration_domain_; }
  
  /// Add a statement to the model
  void addStatement(const PolyhedralStatement& stmt);
  
  /// Add array access information
  void addArrayAccess(const ArrayAccessInfo& access);
  
  /// Build the schedule tree from statements
  void buildScheduleTree();
  
  /// Validate the polyhedral model
  bool validate() const;

private:
  isl_ctx* ctx_;
  isl_schedule* schedule_;
  isl_union_set* iteration_domain_;
  
  std::vector<PolyhedralStatement> statements_;
  std::vector<ArrayAccessInfo> array_accesses_;
  
  /// Internal helper methods
  void computeIterationDomain();
  void buildInitialSchedule();
};

/// Main extractor class for converting MLIR affine to polyhedral model
class PolyhedralExtractor {
public:
  /// Constructor with ISL context
  explicit PolyhedralExtractor(isl_ctx* ctx);
  
  /// Destructor
  ~PolyhedralExtractor();
  
  /// Extract polyhedral model from a single affine for loop
  std::unique_ptr<PolyhedralModel> extractFromAffineFor(mlir::affine::AffineForOp forOp);
  
  /// Extract polyhedral model from a nest of affine operations
  std::unique_ptr<PolyhedralModel> extractFromAffineNest(
      const std::vector<mlir::affine::AffineForOp>& forOps);
  
  /// Extract polyhedral model from a region containing affine operations
  std::unique_ptr<PolyhedralModel> extractFromRegion(mlir::Region& region);
  
  /// Extract polyhedral model from an entire function
  std::unique_ptr<PolyhedralModel> extractFromFunction(mlir::func::FuncOp funcOp);

private:
  isl_ctx* ctx_;
  
  /// Current extraction context
  llvm::DenseMap<mlir::Value, std::string> value_to_name_;
  llvm::DenseMap<mlir::Value, int> loop_depth_;
  std::vector<mlir::affine::AffineForOp> loop_stack_;
  std::map<std::string, const analysis::PolyhedralStatement*> statement_map_;
  
  /// Helper methods for extraction
  void visitOperation(mlir::Operation* op, PolyhedralModel& model);
  void visitAffineFor(mlir::affine::AffineForOp forOp, PolyhedralModel& model);
  void visitAffineIf(mlir::affine::AffineIfOp ifOp, PolyhedralModel& model);
  void visitAffineLoad(mlir::affine::AffineLoadOp loadOp, PolyhedralModel& model);
  void visitAffineStore(mlir::affine::AffineStoreOp storeOp, PolyhedralModel& model);
  void visitAffineYield(mlir::affine::AffineYieldOp yieldOp, PolyhedralModel& model);
  
  /// Convert MLIR affine expressions to ISL
  isl_set* convertAffineDomain(mlir::affine::AffineForOp forOp);
  isl_map* convertAffineMap(mlir::AffineMap affineMap, 
                           const std::vector<mlir::Value>& operands);
  
  /// Create statement domains and access maps
  PolyhedralStatement createStatement(mlir::Operation* op);
  ArrayAccessInfo createArrayAccess(mlir::Value array, mlir::AffineMap accessMap,
                                   bool isRead, mlir::Operation* context);
  
  /// Name generation utilities
  std::string generateStatementName(mlir::Operation* op);
  std::string generateArrayName(mlir::Value array);
  std::string generateValueName(mlir::Value value);
  
  /// Analysis utilities
  bool isAffineAccessPattern(mlir::Operation* op);
  std::vector<int> analyzeStridePattern(mlir::AffineMap accessMap);
  bool hasTemporalReuse(const std::vector<isl_map*>& accesses);
  bool hasSpatialReuse(const std::vector<isl_map*>& accesses);
};

/// Utility functions for polyhedral analysis
class PolyhedralUtils {
public:
  /// Create an ISL context with appropriate options
  static isl_ctx* createContext();
  
  /// Destroy an ISL context
  static void destroyContext(isl_ctx* ctx);
  
  /// Convert MLIR affine map to ISL map
  static isl_map* convertAffineMapToISL(mlir::AffineMap affineMap,
                                       isl_ctx* ctx);
  
  /// Convert MLIR integer set to ISL set
  static isl_set* convertIntegerSetToISL(mlir::IntegerSet intSet,
                                        isl_ctx* ctx);
  
  /// Check if an operation is suitable for polyhedral analysis
  static bool isSuitableForPolyhedral(mlir::Operation* op);
  
  /// Get the affine loop nest containing an operation
  static std::vector<mlir::affine::AffineForOp> getAffineLoopNest(mlir::Operation* op);
  
  /// Analyze memory access patterns in an operation
  static std::vector<mlir::Value> getMemoryAccesses(mlir::Operation* op);
  
  /// Check if all operations in a region are affine
  static bool isRegionAffine(mlir::Region& region);
  
  /// Get the maximum loop depth in a region
  static int getMaxLoopDepth(mlir::Region& region);
  
  /// Check if function has affine loops
  static bool hasAffineLoops(mlir::func::FuncOp funcOp);
  
  /// Get loop depth of a specific loop
  static int getLoopDepth(mlir::affine::AffineForOp forOp);
  
  /// Get nested loops within a loop
  static std::vector<mlir::affine::AffineForOp> getNestedLoops(mlir::affine::AffineForOp forOp);
  
  /// Check if loop nest is perfect
  static bool isPerfectLoopNest(mlir::affine::AffineForOp forOp);
  
  /// Get statements within a loop
  static std::vector<mlir::Operation*> getStatements(mlir::affine::AffineForOp forOp);
  
  /// Check if function has complex control flow
  static bool hasComplexControl(mlir::func::FuncOp funcOp);
  
  /// Get operation signature string
  static std::string getOperationSignature(mlir::Operation* op);
  
  /// Check if polyhedral model can be extracted from function
  static bool canExtractPolyhedralModel(mlir::func::FuncOp funcOp);
};

} // namespace analysis
} // namespace autopoly

#endif // AUTOPOLY_ANALYSIS_POLYHEDRALEXTRACTION_H
