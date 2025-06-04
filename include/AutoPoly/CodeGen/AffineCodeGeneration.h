//===- AffineCodeGeneration.h - Polyhedral to Affine Codegen -*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file defines code generation from polyhedral schedules back to
// MLIR affine dialect operations, preserving parallel and optimization
// annotations for downstream compilation.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOPOLY_CODEGEN_AFFINECODEGEN_H
#define AUTOPOLY_CODEGEN_AFFINECODEGEN_H

#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "AutoPoly/Transform/SchedulingTransform.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include <memory>
#include <vector>
#include <map>

// Forward declarations for ISL structures
struct isl_ctx;
struct isl_schedule;
struct isl_schedule_node;
struct isl_ast_node;
struct isl_ast_build;
struct isl_ast_expr;

namespace autopoly {
namespace codegen {

/// Code generation options for controlling output format
struct CodeGenOptions {
  /// Loop generation options
  bool generate_parallel_loops = true;      ///< Use affine.parallel for parallel loops
  bool generate_vector_hints = true;        ///< Add vectorization hints
  bool preserve_debug_info = true;          ///< Preserve original debug information
  bool add_performance_annotations = true;  ///< Add performance-related attributes
  
  /// Memory access optimization
  bool optimize_memory_accesses = true;     ///< Optimize affine memory accesses
  bool use_prefetch_hints = false;          ///< Add prefetch hints where beneficial
  bool enable_memory_coalescing = true;     ///< Optimize for memory coalescing
  
  /// Target-specific code generation
  bool target_specific_optimizations = true; ///< Apply target-specific optimizations
  std::string target_triple = "";           ///< Target triple for code generation
  
  /// Debug and verification
  bool verify_generated_code = true;        ///< Verify generated affine operations
  bool dump_intermediate_ast = false;       ///< Dump ISL AST for debugging
  bool add_transformation_metadata = true;  ///< Add metadata about transformations
};

/// Information about generated loop structures
struct GeneratedLoopInfo {
  mlir::affine::AffineForOp for_op;                 ///< Generated affine.for operation
  std::vector<mlir::Value> induction_vars;  ///< Induction variables
  bool is_parallel = false;                 ///< Whether loop is parallel
  bool is_vectorizable = false;             ///< Whether loop is vectorizable
  bool is_unrolled = false;                 ///< Whether loop should be unrolled
  
  /// Tiling information
  bool is_tiled = false;                    ///< Whether loop is result of tiling
  int tile_size = 0;                        ///< Tile size if tiled
  mlir::affine::AffineForOp outer_tile_loop;        ///< Outer tile loop (if tiled)
  
  /// Performance hints
  std::map<std::string, mlir::Attribute> performance_hints;
};

/// Result of affine code generation
struct CodeGenResult {
  mlir::Operation* root_operation;          ///< Root of generated code
  std::vector<GeneratedLoopInfo> loop_info; ///< Information about generated loops
  bool generation_successful = false;       ///< Whether generation succeeded
  std::string error_message;                ///< Error message if failed
  
  /// Statistics
  int loops_generated = 0;                  ///< Number of loops generated
  int parallel_loops = 0;                   ///< Number of parallel loops
  int vectorizable_loops = 0;               ///< Number of vectorizable loops
  int memory_accesses_optimized = 0;        ///< Number of optimized memory accesses
  
  /// Verification results
  bool passes_verification = true;          ///< Generated code passes verification
  std::vector<std::string> verification_warnings; ///< Verification warnings
};

/// Main affine code generator
class AffineCodeGenerator {
public:
  /// Constructor with MLIR context and ISL context
  AffineCodeGenerator(mlir::MLIRContext* mlir_ctx, isl_ctx* isl_ctx);
  
  /// Destructor
  ~AffineCodeGenerator();
  
  /// Generate affine code from polyhedral schedule
  CodeGenResult generateCode(isl_schedule* schedule,
                           const analysis::PolyhedralModel& original_model,
                           const transform::TransformationResult& transform_result,
                           mlir::OpBuilder& builder,
                           mlir::Location loc);
  
  /// Generate code with specific options
  CodeGenResult generateCodeWithOptions(isl_schedule* schedule,
                                      const analysis::PolyhedralModel& original_model,
                                      const transform::TransformationResult& transform_result,
                                      mlir::OpBuilder& builder,
                                      mlir::Location loc,
                                      const CodeGenOptions& options);
  
  /// Replace original affine operations with transformed ones
  bool replaceOriginalCode(mlir::Operation* original_root,
                          const CodeGenResult& generated_code);

private:
  mlir::MLIRContext* mlir_ctx_;
  isl_ctx* isl_ctx_;
  CodeGenOptions default_options_;
  
  /// ISL AST generation
  isl_ast_node* generateISLAST(isl_schedule* schedule);
  isl_ast_build* createASTBuild(const analysis::PolyhedralModel& model);
  
  /// MLIR code generation from ISL AST
  mlir::Operation* generateFromASTNode(isl_ast_node* node,
                                     mlir::OpBuilder& builder,
                                     mlir::Location loc,
                                     CodeGenResult& result);
  
  /// Specific AST node handlers
  mlir::affine::AffineForOp generateForLoop(isl_ast_node* for_node,
                                  mlir::OpBuilder& builder,
                                  mlir::Location loc,
                                  CodeGenResult& result);
  
  mlir::affine::AffineIfOp generateIfCondition(isl_ast_node* if_node,
                                     mlir::OpBuilder& builder,
                                     mlir::Location loc,
                                     CodeGenResult& result);
  
  mlir::Operation* generateBlock(isl_ast_node* block_node,
                               mlir::OpBuilder& builder,
                               mlir::Location loc,
                               CodeGenResult& result);
  
  mlir::Operation* generateUser(isl_ast_node* user_node,
                              mlir::OpBuilder& builder,
                              mlir::Location loc,
                              CodeGenResult& result);
  
  /// Statement generation
  mlir::Operation* generateStatement(const std::string& statement_name,
                                   const std::vector<mlir::Value>& iteration_values,
                                   const analysis::PolyhedralModel& original_model,
                                   mlir::OpBuilder& builder,
                                   mlir::Location loc);
  
  /// Memory access generation
  mlir::Value generateMemoryAccess(mlir::Value array,
                                 const std::vector<mlir::Value>& indices,
                                 bool is_load,
                                 mlir::OpBuilder& builder,
                                 mlir::Location loc);
  
  /// Parallel loop generation
  mlir::affine::AffineParallelOp generateParallelLoop(isl_ast_node* for_node,
                                            mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            CodeGenResult& result);
  
  /// Vectorization hint generation
  void addVectorizationHints(mlir::affine::AffineForOp forOp,
                           const GeneratedLoopInfo& loop_info);
  
  /// Performance annotation generation
  void addPerformanceAnnotations(mlir::Operation* op,
                               const transform::TransformationResult& transform_result);
  
  /// Utility methods
  mlir::AffineMap convertISLMapToAffineMap(isl_map* isl_map);
  mlir::IntegerSet convertISLSetToIntegerSet(isl_set* isl_set);
  mlir::Value createAffineExpression(isl_ast_expr* expr,
                                   mlir::OpBuilder& builder,
                                   mlir::Location loc);
  
  /// Value mapping between ISL and MLIR
  std::map<std::string, mlir::Value> value_map_;
  std::map<std::string, const analysis::PolyhedralStatement*> statement_map_;
};

/// Parallel loop optimization
class ParallelLoopOptimizer {
public:
  /// Constructor
  explicit ParallelLoopOptimizer(mlir::MLIRContext* ctx);
  
  /// Optimize parallel loops for target architecture
  void optimizeParallelLoops(mlir::Operation* root,
                           const target::TargetCharacteristics& target);
  
  /// Convert serial loops to parallel where beneficial
  bool convertToParallel(mlir::affine::AffineForOp forOp,
                        const target::TargetCharacteristics& target);
  
  /// Optimize nested parallelism
  void optimizeNestedParallelism(mlir::affine::AffineParallelOp parallelOp,
                               const target::TargetCharacteristics& target);

private:
  mlir::MLIRContext* ctx_;
  
  /// Analysis methods
  bool isParallelizationBeneficial(mlir::affine::AffineForOp forOp,
                                 const target::TargetCharacteristics& target);
  
  int estimateParallelWorkload(mlir::affine::AffineForOp forOp);
  bool hasMemoryDependencies(mlir::affine::AffineForOp forOp);
};

/// Memory access optimizer
class MemoryAccessOptimizer {
public:
  /// Constructor
  explicit MemoryAccessOptimizer(mlir::MLIRContext* ctx);
  
  /// Optimize memory access patterns
  void optimizeMemoryAccesses(mlir::Operation* root,
                            const target::TargetCharacteristics& target);
  
  /// Add prefetch hints
  void addPrefetchHints(mlir::Operation* root,
                       const target::TargetCharacteristics& target);
  
  /// Optimize for memory coalescing
  void optimizeForCoalescing(mlir::Operation* root,
                           const target::TargetCharacteristics& target);

private:
  mlir::MLIRContext* ctx_;
  
  /// Analysis methods
  bool isPrefetchBeneficial(mlir::affine::AffineLoadOp loadOp,
                          const target::TargetCharacteristics& target);
  
  bool isCoalescedAccess(mlir::affine::AffineLoadOp loadOp);
  void insertPrefetch(mlir::affine::AffineLoadOp loadOp, int distance);
};

/// Code verification and validation
class CodeVerifier {
public:
  /// Verify generated affine code
  static bool verifyAffineCode(mlir::Operation* root);
  
  /// Check semantic equivalence with original code
  static bool checkSemanticEquivalence(mlir::Operation* original,
                                     mlir::Operation* generated);
  
  /// Validate memory access bounds
  static bool validateMemoryAccesses(mlir::Operation* root);
  
  /// Check for race conditions in parallel code
  static bool checkRaceConditions(mlir::Operation* root);
  
  /// Verify transformation correctness
  static bool verifyTransformationCorrectness(
      const analysis::PolyhedralModel& original_model,
      const CodeGenResult& generated_code);
};

/// Utility functions for affine code generation
class CodeGenUtils {
public:
  /// Convert ISL expression to MLIR affine expression
  static mlir::AffineExpr convertISLExprToAffine(isl_ast_expr* expr,
                                               mlir::MLIRContext* ctx);
  
  /// Create loop bounds from ISL constraints
  static std::pair<mlir::AffineMap, mlir::AffineMap> createLoopBounds(
      isl_ast_node* for_node, mlir::MLIRContext* ctx);
  
  /// Extract parallel loop dimensions from schedule
  static std::vector<int> extractParallelDimensions(isl_schedule* schedule);
  
  /// Generate unique names for generated operations
  static std::string generateUniqueName(const std::string& base);
  
  /// Add debug information to generated operations
  static void addDebugInfo(mlir::Operation* op, const std::string& info);
  
  /// Create performance attributes
  static mlir::DictionaryAttr createPerformanceAttrs(
      mlir::MLIRContext* ctx,
      const std::map<std::string, int>& params);
};

} // namespace codegen
} // namespace autopoly

#endif // AUTOPOLY_CODEGEN_AFFINECODEGEN_H
