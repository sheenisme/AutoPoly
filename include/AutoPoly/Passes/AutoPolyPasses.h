//===- AutoPolyPasses.h - AutoPoly MLIR Passes ----------------*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file defines MLIR passes for the AutoPoly polyhedral scheduling
// framework, providing automatic scheduling transformations for affine
// dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOPOLY_PASSES_AUTOPOLYPASSES_H
#define AUTOPOLY_PASSES_AUTOPOLYPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <memory>
#include <string>

namespace autopoly {
namespace passes {

/// Pass options for AutoPoly scheduling transformation
struct AutoPolyPassOptions {
  /// Target specification
  std::string target_type = "auto";         ///< Target hardware type (auto-detect if "auto")
  std::string target_name = "";             ///< Specific target name (optional)
  
  /// Scheduling algorithm selection
  std::string scheduling_algorithm = "auto"; ///< Scheduling algorithm to use
  std::string scheduling_strategy = "auto";  ///< Scheduling strategy to use
  
  /// Optimization technique selection
  bool enable_tiling = true;                ///< Enable loop tiling
  bool enable_fusion = true;                ///< Enable loop fusion
  bool enable_skewing = false;              ///< Enable loop skewing
  bool enable_parallelization = true;       ///< Enable parallelization
  bool enable_vectorization = true;         ///< Enable vectorization hints
  bool enable_unrolling = false;            ///< Enable loop unrolling
  
  /// Transformation parameters
  std::vector<int> tile_sizes;              ///< Explicit tile sizes (empty = auto)
  std::vector<int> unroll_factors;          ///< Explicit unroll factors (empty = auto)
  int max_parallel_depth = 3;               ///< Maximum parallel nesting depth
  
  /// Debug and analysis options
  bool debug_mode = false;                  ///< Enable debug output
  bool verify_transformations = true;       ///< Verify transformation correctness
  bool dump_polyhedral_model = false;       ///< Dump extracted polyhedral model
  bool dump_dependences = false;            ///< Dump dependence analysis results
  bool dump_schedules = false;              ///< Dump scheduling results
  
  /// Performance tuning
  int scheduling_timeout = 60;              ///< Timeout for scheduling (seconds)
  bool enable_aggressive_optimization = false; ///< Enable aggressive optimizations
  double min_speedup_threshold = 1.1;       ///< Minimum speedup to apply transformation
};

/// Main AutoPoly scheduling pass
class AutoPolySchedulingPass : public mlir::OperationPass<mlir::func::FuncOp> {
public:
  /// Constructor with default options
  AutoPolySchedulingPass();
  
  /// Constructor with specific options
  explicit AutoPolySchedulingPass(const AutoPolyPassOptions& options);
  
  /// Copy constructor for pass cloning
  AutoPolySchedulingPass(const AutoPolySchedulingPass& other);
  
  /// Get pass argument (for registration)
  llvm::StringRef getArgument() const override { return "autopoly-scheduling"; }
  
  /// Get pass name
  llvm::StringRef getName() const override { return "autopoly-scheduling"; }
  
  /// Get pass description
  llvm::StringRef getDescription() const override {
    return "AutoPoly polyhedral scheduling transformation pass";
  }
  
  /// Clone the pass
  std::unique_ptr<mlir::Pass> clonePass() const override;
  
  /// Get pass dependent dialects
  void getDependentDialects(mlir::DialectRegistry& registry) const override;

protected:
  /// Main pass execution
  void runOnOperation() override;

private:
  AutoPolyPassOptions options_;
  
  /// Pass implementation methods
  bool analyzeFunction(mlir::func::FuncOp funcOp);
  bool transformFunction(mlir::func::FuncOp funcOp);
  void generateStatistics();
  
  /// Utility methods
  bool isAffineRegionSuitable(mlir::Region& region);
  void reportTransformationResults(const std::string& results);
  void dumpDebugInformation(mlir::func::FuncOp funcOp);
};

/// Pass for extracting and analyzing polyhedral models
class PolyhedralAnalysisPass : public mlir::OperationPass<mlir::func::FuncOp> {
public:
  PolyhedralAnalysisPass();
  
  llvm::StringRef getName() const override { return "polyhedral-analysis"; }
  
  llvm::StringRef getDescription() const override {
    return "Extract and analyze polyhedral models from affine operations";
  }
  
  std::unique_ptr<mlir::Pass> clonePass() const override;
  
  void getDependentDialects(mlir::DialectRegistry& registry) const override;

protected:
  void runOnOperation() override;

private:
  bool enable_debug_output_ = false;
  
  void analyzeAffineRegions(mlir::func::FuncOp funcOp);
  void dumpPolyhedralModel(mlir::func::FuncOp funcOp);
};

/// Pass for dependence analysis
class DependenceAnalysisPass : public mlir::OperationPass<mlir::func::FuncOp> {
public:
  DependenceAnalysisPass();
  
  llvm::StringRef getName() const override { return "dependence-analysis"; }
  
  llvm::StringRef getDescription() const override {
    return "Perform polyhedral dependence analysis on affine operations";
  }
  
  std::unique_ptr<mlir::Pass> clonePass() const override;
  
  void getDependentDialects(mlir::DialectRegistry& registry) const override;

protected:
  void runOnOperation() override;

private:
  bool enable_debug_output_ = false;
  
  void analyzeDependences(mlir::func::FuncOp funcOp);
  void dumpDependenceInformation(mlir::func::FuncOp funcOp);
};

/// Pass for target detection and characterization
class TargetDetectionPass : public mlir::OperationPass<mlir::ModuleOp> {
public:
  TargetDetectionPass();
  
  llvm::StringRef getName() const override { return "target-detection"; }
  
  llvm::StringRef getDescription() const override {
    return "Detect and characterize available target hardware";
  }
  
  std::unique_ptr<mlir::Pass> clonePass() const override;

protected:
  void runOnOperation() override;

private:
  void detectTargets(mlir::ModuleOp moduleOp);
  void reportTargetInformation();
};

/// Pass pipeline builder for AutoPoly
class AutoPolyPipelineBuilder {
public:
  /// Add AutoPoly passes to a pass manager
  static void addAutoPolyPasses(mlir::PassManager& pm,
                               const AutoPolyPassOptions& options = {});
  
  /// Add analysis-only passes
  static void addAnalysisPasses(mlir::PassManager& pm);
  
  /// Add transformation passes
  static void addTransformationPasses(mlir::PassManager& pm,
                                    const AutoPolyPassOptions& options = {});
  
  /// Create a complete AutoPoly pipeline
  static std::unique_ptr<mlir::PassManager> createPipeline(
      mlir::MLIRContext* context,
      const AutoPolyPassOptions& options = {});
};

/// Pass registration functions
void registerAutoPolyPasses();

/// Factory functions for creating passes
std::unique_ptr<mlir::Pass> createAutoPolySchedulingPass();
std::unique_ptr<mlir::Pass> createAutoPolySchedulingPass(const AutoPolyPassOptions& options);
std::unique_ptr<mlir::Pass> createPolyhedralAnalysisPass();
std::unique_ptr<mlir::Pass> createDependenceAnalysisPass();
std::unique_ptr<mlir::Pass> createTargetDetectionPass();

/// Command line option parsing for passes
template <typename Options>
void addAutoPolyPassOptions(mlir::PassPipelineRegistration<Options>& registration);

/// Pass statistics and metrics
struct AutoPolyPassStatistics {
  /// Analysis statistics
  int functions_analyzed = 0;
  int affine_loops_found = 0;
  int polyhedral_models_extracted = 0;
  int dependence_relations_computed = 0;
  
  /// Transformation statistics
  int functions_transformed = 0;
  int loops_tiled = 0;
  int loops_parallelized = 0;
  int loops_fused = 0;
  int loops_skewed = 0;
  
  /// Performance statistics
  double total_analysis_time = 0.0;
  double total_transformation_time = 0.0;
  double estimated_total_speedup = 1.0;
  
  /// Target statistics
  std::string detected_target_type;
  int available_targets = 0;
  
  /// Error statistics
  int analysis_failures = 0;
  int transformation_failures = 0;
  
  /// Reset all statistics
  void reset();
  
  /// Convert to string for reporting
  std::string toString() const;
};

/// Global statistics instance
extern AutoPolyPassStatistics g_pass_statistics;

} // namespace passes
} // namespace autopoly

#endif // AUTOPOLY_PASSES_AUTOPOLYPASSES_H
