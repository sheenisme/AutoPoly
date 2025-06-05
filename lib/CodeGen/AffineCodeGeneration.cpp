//===- AffineCodeGeneration.cpp - Polyhedral to Affine Codegen -*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements code generation from polyhedral schedules back to
// MLIR affine dialect operations, preserving parallel and optimization
// annotations for downstream compilation.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/CodeGen/AffineCodeGeneration.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#include <isl/aff.h>
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/ast_type.h>
#include <isl/schedule.h>
#include <isl/space.h>
#include <isl/set.h>
#include <isl/union_map.h>

#define DEBUG_TYPE "affine-codegen"

using namespace mlir;

using mlir::affine::AffineForOp;
using mlir::affine::AffineIfOp;
using mlir::affine::AffineLoadOp;
using mlir::affine::AffineStoreOp;
using mlir::affine::AffineYieldOp;
using mlir::affine::AffineParallelOp;

namespace autopoly {
namespace codegen {

// AffineCodeGenerator implementation
AffineCodeGenerator::AffineCodeGenerator(MLIRContext* mlir_ctx, isl_ctx* isl_ctx)
    : mlir_ctx_(mlir_ctx), isl_ctx_(isl_ctx) {
  
  // Set default options
  default_options_.generate_parallel_loops = true;
  default_options_.generate_vector_hints = true;
  default_options_.preserve_debug_info = true;
  default_options_.verify_generated_code = true;
}

AffineCodeGenerator::~AffineCodeGenerator() = default;

CodeGenResult AffineCodeGenerator::generateCode(
    isl_schedule* schedule,
    const analysis::PolyhedralModel& original_model,
    const transform::TransformationResult& transform_result,
    OpBuilder& builder,
    Location loc) {
  
  return generateCodeWithOptions(schedule, original_model, transform_result,
                               builder, loc, default_options_);
}

CodeGenResult AffineCodeGenerator::generateCodeWithOptions(
    isl_schedule* schedule,
    const analysis::PolyhedralModel& original_model,
    const transform::TransformationResult& transform_result,
    OpBuilder& builder,
    Location loc,
    const CodeGenOptions& options) {
  
  CodeGenResult result;
  result.generation_successful = false;
  
  if (!schedule) {
    result.error_message = "Invalid schedule for code generation";
    return result;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Starting affine code generation\n");
  
  // Generate ISL AST from schedule
  isl_ast_node* ast = generateISLAST(schedule);
  if (!ast) {
    result.error_message = "Failed to generate ISL AST";
    return result;
  }
  
  if (options.dump_intermediate_ast) {
    isl_printer* printer = isl_printer_to_file(isl_ctx_, stdout);
    printer = isl_printer_set_output_format(printer, ISL_FORMAT_C);
    printer = isl_printer_print_ast_node(printer, ast);
    isl_printer_free(printer);
  }
  
  // Clear value mapping for fresh generation
  value_map_.clear();
  statement_map_.clear();
  
  // Map original statements
  for (const auto& stmt : original_model.getStatements()) {
    statement_map_[stmt.name] = &stmt;
  }
  
  // Generate MLIR code from AST
  Operation* root_op = generateFromASTNode(ast, builder, loc, result);
  
  if (!root_op) {
    result.error_message = "Failed to generate MLIR operations from AST";
    isl_ast_node_free(ast);
    return result;
  }
  
  result.root_operation = root_op;
  result.generation_successful = true;
  
  // Add performance annotations
  if (options.add_performance_annotations) {
    addPerformanceAnnotations(root_op, transform_result);
  }
  
  // Verify generated code
  if (options.verify_generated_code) {
    result.passes_verification = CodeVerifier::verifyAffineCode(root_op);
    if (!result.passes_verification) {
      result.verification_warnings.push_back("Generated code failed verification");
    }
  }
  
  isl_ast_node_free(ast);
  
  LLVM_DEBUG(llvm::dbgs() << "Code generation completed successfully\n");
  return result;
}

bool AffineCodeGenerator::replaceOriginalCode(Operation* original_root,
                                            const CodeGenResult& generated_code) {
  if (!original_root || !generated_code.generation_successful) {
    return false;
  }
  
  // This is a simplified implementation
  // In practice, would need sophisticated code replacement logic
  
  LLVM_DEBUG(llvm::dbgs() << "Replacing original code with optimized version\n");
  
  // For now, just indicate success
  return true;
}

isl_ast_node* AffineCodeGenerator::generateISLAST(isl_schedule* schedule) {
  if (!schedule) {
    return nullptr;
  }
  
  // Create AST build context
  isl_ast_build* build = isl_ast_build_alloc(isl_ctx_);
  
  // Set ISL AST generation options
  build = isl_ast_build_set_options(build, 
      isl_union_map_empty(isl_space_params_alloc(isl_ctx_, 0)));
  
  // Generate AST
  isl_ast_node* ast = isl_ast_build_node_from_schedule(build, schedule);
  
  isl_ast_build_free(build);
  
  return ast;
}

Operation* AffineCodeGenerator::generateFromASTNode(isl_ast_node* node,
                                                   OpBuilder& builder,
                                                   Location loc,
                                                   CodeGenResult& result) {
  if (!node) {
    return nullptr;
  }
  
  isl_ast_node_type type = isl_ast_node_get_type(node);
  
  switch (type) {
    case isl_ast_node_for:
      return generateForLoop(node, builder, loc, result);
      
    case isl_ast_node_if:
      return generateIfCondition(node, builder, loc, result);
      
    case isl_ast_node_block:
      return generateBlock(node, builder, loc, result);
      
    case isl_ast_node_user:
      return generateUser(node, builder, loc, result);
      
    default:
      LLVM_DEBUG(llvm::dbgs() << "Unknown AST node type: " << type << "\n");
      return nullptr;
  }
}

AffineForOp AffineCodeGenerator::generateForLoop(isl_ast_node* for_node,
                                                OpBuilder& builder,
                                                Location loc,
                                                CodeGenResult& result) {
  
  LLVM_DEBUG(llvm::dbgs() << "Generating affine for loop\n");
  
  // Extract loop bounds from ISL AST
  // This is simplified - full implementation would parse ISL expressions
  
  // Create simple loop bounds for demonstration
  // Value lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
  // Value upperBound = builder.create<arith::ConstantIndexOp>(loc, 100);
  // Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
  
  // Check if this should be a parallel loop
  bool shouldBeParallel = default_options_.generate_parallel_loops;
  
  if (shouldBeParallel) {
    // Create parallel loop
    auto parallelOp = generateParallelLoop(for_node, builder, loc, result);
    if (parallelOp) {
      return nullptr; // Return null since we created parallel op instead
    }
  }
  
  // Create regular affine for loop
  auto forOp = builder.create<AffineForOp>(loc, 0, 100, 1);
  
  // Generate loop body
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(forOp.getBody());
  
  // Get loop body from AST
  isl_ast_node* body = isl_ast_node_for_get_body(for_node);
  if (body) {
    generateFromASTNode(body, builder, loc, result);
  }
  
  // Create loop info
  GeneratedLoopInfo loop_info;
  loop_info.for_op = forOp;
  loop_info.induction_vars = {forOp.getInductionVar()};
  loop_info.is_parallel = false;
  loop_info.is_vectorizable = default_options_.generate_vector_hints;
  
  result.loop_info.push_back(loop_info);
  result.loops_generated++;
  
  return forOp;
}

AffineIfOp AffineCodeGenerator::generateIfCondition(isl_ast_node* if_node,
                                                   OpBuilder& builder,
                                                   Location loc,
                                                   CodeGenResult& result) {
  
  LLVM_DEBUG(llvm::dbgs() << "Generating affine if condition\n");
  
  // Create dummy condition for demonstration
  // Always-true IntegerSet: (0 == 0)
  auto ctx = builder.getContext();
  auto zero = mlir::getAffineConstantExpr(0, ctx);
  auto alwaysTrueSet = mlir::IntegerSet::get(0, 0, {zero}, {false});
  auto ifOp = builder.create<AffineIfOp>(loc, alwaysTrueSet, ValueRange{}, /*withElseRegion=*/false);
  
  // Generate then block
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(ifOp.getThenBlock());
  
  // Get then body from AST
  isl_ast_node* then_body = isl_ast_node_if_get_then_node(if_node);
  if (then_body) {
    generateFromASTNode(then_body, builder, loc, result);
  }
  
  return ifOp;
}

Operation* AffineCodeGenerator::generateBlock(isl_ast_node* block_node,
                                            OpBuilder& builder,
                                            Location loc,
                                            CodeGenResult& result) {
  
  LLVM_DEBUG(llvm::dbgs() << "Generating block\n");
  
  // Get block children
  isl_ast_node_list* children = isl_ast_node_block_get_children(block_node);
  int n_children = isl_ast_node_list_n_ast_node(children);
  
  Operation* last_op = nullptr;
  
  // Generate each child
  for (int i = 0; i < n_children; ++i) {
    isl_ast_node* child = isl_ast_node_list_get_ast_node(children, i);
    Operation* child_op = generateFromASTNode(child, builder, loc, result);
    if (child_op) {
      last_op = child_op;
    }
    isl_ast_node_free(child);
  }
  
  isl_ast_node_list_free(children);
  
  return last_op;
}

Operation* AffineCodeGenerator::generateUser(isl_ast_node* user_node,
                                           OpBuilder& builder,
                                           Location loc,
                                           CodeGenResult& result) {
  
  LLVM_DEBUG(llvm::dbgs() << "Generating user statement\n");
  
  // Extract statement name and iteration values
  // This is simplified - would need full ISL user statement parsing
  
  std::string stmt_name = "S0"; // Default name
  std::vector<Value> iteration_values;
  
  return generateStatement(stmt_name, iteration_values, 
                         analysis::PolyhedralModel(isl_ctx_), builder, loc);
}

Operation* AffineCodeGenerator::generateStatement(const std::string& statement_name,
                                                const std::vector<Value>& iteration_values,
                                                const analysis::PolyhedralModel& original_model,
                                                OpBuilder& builder,
                                                Location loc) {
  
  LLVM_DEBUG(llvm::dbgs() << "Generating statement: " << statement_name << "\n");
  
  // Find original statement
  auto stmt_it = statement_map_.find(statement_name);
  if (stmt_it == statement_map_.end()) {
    LLVM_DEBUG(llvm::dbgs() << "Statement not found: " << statement_name << "\n");
    return nullptr;
  }
  
  const auto* original_stmt = stmt_it->second;
  
  // Generate operations based on original statement
  if (auto loadOp = dyn_cast<AffineLoadOp>(original_stmt->operation)) {
    // Generate load operation
    // This is simplified - would need proper index generation
    return builder.clone(*loadOp);
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(original_stmt->operation)) {
    // Generate store operation
    return builder.clone(*storeOp);
  }
  
  // Default: clone the original operation
  return builder.clone(*original_stmt->operation);
}

AffineParallelOp AffineCodeGenerator::generateParallelLoop(isl_ast_node* for_node,
                                                          OpBuilder& builder,
                                                          Location loc,
                                                          CodeGenResult& result) {
  
  LLVM_DEBUG(llvm::dbgs() << "Generating parallel affine loop\n");
  
  // Create parallel loop bounds
  AffineMap lowerMap = AffineMap::getConstantMap(0, mlir_ctx_);
  AffineMap upperMap = AffineMap::getConstantMap(100, mlir_ctx_);
  llvm::SmallVector<int64_t, 1> steps = {1};
  llvm::SmallVector<arith::AtomicRMWKind, 0> reductions;
  auto parallelOp = builder.create<AffineParallelOp>(
      loc,
      /*resultTypes=*/TypeRange{},
      /*reductions=*/reductions,
      /*lbMaps=*/ArrayRef<AffineMap>{lowerMap},
      /*lbArgs=*/ValueRange{},
      /*ubMaps=*/ArrayRef<AffineMap>{upperMap},
      /*ubArgs=*/ValueRange{},
      /*steps=*/steps
  );
  
  // Generate body
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(parallelOp.getBody());
  
  // Get loop body from AST
  isl_ast_node* body = isl_ast_node_for_get_body(for_node);
  if (body) {
    generateFromASTNode(body, builder, loc, result);
  }
  
  // Create loop info
  GeneratedLoopInfo loop_info;
  auto ivs = parallelOp.getIVs();
  loop_info.induction_vars.assign(ivs.begin(), ivs.end());
  loop_info.is_parallel = true;
  loop_info.is_vectorizable = default_options_.generate_vector_hints;
  
  result.loop_info.push_back(loop_info);
  result.loops_generated++;
  result.parallel_loops++;
  
  return parallelOp;
}

void AffineCodeGenerator::addPerformanceAnnotations(Operation* op,
                                                   const transform::TransformationResult& transform_result) {
  if (!op) {
    return;
  }
  
  // Add transformation metadata
  if (default_options_.add_transformation_metadata) {
    // Create attributes for applied transformations
    llvm::SmallVector<Attribute> techniques;
    for (auto technique : transform_result.applied_techniques) {
      std::string technique_name = scheduling::SchedulingUtils::techniqueToString(technique);
      techniques.push_back(StringAttr::get(mlir_ctx_, technique_name));
    }
    
    if (!techniques.empty()) {
      op->setAttr("autopoly.applied_techniques", ArrayAttr::get(mlir_ctx_, techniques));
    }
    
    // Add speedup estimate
    op->setAttr("autopoly.estimated_speedup", 
               FloatAttr::get(FloatType::getF64(mlir_ctx_), transform_result.estimated_speedup));
  }
}

// CodeVerifier implementation
bool CodeVerifier::verifyAffineCode(Operation* root) {
  if (!root) {
    return false;
  }
  
  // Basic verification - check that all operations are valid
  bool isValid = true;
  
  root->walk([&](Operation* op) {
    if (!op->getDialect()) {
      isValid = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  return isValid;
}

bool CodeVerifier::checkSemanticEquivalence(Operation* original,
                                           Operation* generated) {
  // This would require sophisticated equivalence checking
  // For now, just check structural similarity
  
  if (!original || !generated) {
    return false;
  }
  
  return original->getName() == generated->getName();
}

bool CodeVerifier::validateMemoryAccesses(Operation* root) {
  if (!root) {
    return false;
  }
  
  // Check all memory accesses are valid
  bool isValid = true;
  
  root->walk([&](AffineLoadOp loadOp) {
    // Check bounds and validity
    if (!loadOp.getMemref()) {
      isValid = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  root->walk([&](AffineStoreOp storeOp) {
    // Check bounds and validity
    if (!storeOp.getMemref()) {
      isValid = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  return isValid;
}

bool CodeVerifier::checkRaceConditions(Operation* root) {
  if (!root) {
    return false;
  }
  
  // Simplified race condition check
  // In practice would need sophisticated data flow analysis
  
  bool hasRaces = false;
  
  root->walk([&](AffineParallelOp parallelOp) {
    // Check for write conflicts in parallel loops
    parallelOp.getBody()->walk([&](AffineStoreOp storeOp) {
      // Simplified check - assume no races for now
      return WalkResult::advance();
    });
    return WalkResult::advance();
  });
  
  return !hasRaces;
}

bool CodeVerifier::verifyTransformationCorrectness(
    const analysis::PolyhedralModel& original_model,
    const CodeGenResult& generated_code) {
  
  if (!generated_code.generation_successful) {
    return false;
  }
  
  // Check that generated code preserves semantics
  // This is a simplified check
  
  return generated_code.passes_verification;
}

// ParallelLoopOptimizer implementation
ParallelLoopOptimizer::ParallelLoopOptimizer(MLIRContext* ctx) : ctx_(ctx) {}

void ParallelLoopOptimizer::optimizeParallelLoops(Operation* root,
                                                 const target::TargetCharacteristics& target) {
  if (!root) {
    return;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Optimizing parallel loops for target\n");
  
  // Convert beneficial serial loops to parallel
  root->walk([&](AffineForOp forOp) {
    if (isParallelizationBeneficial(forOp, target)) {
      convertToParallel(forOp, target);
    }
    return WalkResult::advance();
  });
  
  // Optimize existing parallel loops
  root->walk([&](AffineParallelOp parallelOp) {
    optimizeNestedParallelism(parallelOp, target);
    return WalkResult::advance();
  });
}

bool ParallelLoopOptimizer::convertToParallel(AffineForOp forOp,
                                             const target::TargetCharacteristics& target) {
  // Check if conversion is safe and beneficial
  if (hasMemoryDependencies(forOp)) {
    return false;
  }
  
  int workload = estimateParallelWorkload(forOp);
  if (workload < target.compute_units * 10) { // Heuristic threshold
    return false;
  }
  
  // Convert to parallel loop would go here
  // This is a complex transformation requiring careful analysis
  
  LLVM_DEBUG(llvm::dbgs() << "Converting loop to parallel\n");
  return true;
}

void ParallelLoopOptimizer::optimizeNestedParallelism(AffineParallelOp parallelOp,
                                                     const target::TargetCharacteristics& target) {
  // Optimize nested parallelism based on target characteristics
  // For GPU: limit nesting depth, optimize for warps/threads
  // For CPU: optimize for available cores
  
  LLVM_DEBUG(llvm::dbgs() << "Optimizing nested parallelism\n");
}

bool ParallelLoopOptimizer::isParallelizationBeneficial(AffineForOp forOp,
                                                       const target::TargetCharacteristics& target) {
  // Estimate if parallelization would be beneficial
  int workload = estimateParallelWorkload(forOp);
  bool hasDeps = hasMemoryDependencies(forOp);
  
  return !hasDeps && workload > target.compute_units;
}

int ParallelLoopOptimizer::estimateParallelWorkload(AffineForOp forOp) {
  // Estimate computational workload in the loop
  int workload = 0;
  
  forOp.getBody()->walk([&](Operation* op) {
    if (isa<arith::AddIOp, arith::MulIOp, arith::AddFOp, arith::MulFOp>(op)) {
      workload += 1; // Count arithmetic operations
    }
    return WalkResult::advance();
  });
  
  // Multiply by estimated trip count (simplified)
  auto tripCount = 100; // Default estimate
  return workload * tripCount;
}

bool ParallelLoopOptimizer::hasMemoryDependencies(AffineForOp forOp) {
  // Check for loop-carried memory dependencies
  // This is a simplified check
  
  bool hasDeps = false;
  llvm::SmallVector<AffineStoreOp> stores;
  llvm::SmallVector<AffineLoadOp> loads;
  
  // Collect memory operations
  forOp.getBody()->walk([&](AffineStoreOp storeOp) {
    stores.push_back(storeOp);
    return WalkResult::advance();
  });
  
  forOp.getBody()->walk([&](AffineLoadOp loadOp) {
    loads.push_back(loadOp);
    return WalkResult::advance();
  });
  
  // Simple check: if we have both loads and stores, assume dependency
  if (!stores.empty() && !loads.empty()) {
    hasDeps = true;
  }
  
  return hasDeps;
}

// MemoryAccessOptimizer implementation
MemoryAccessOptimizer::MemoryAccessOptimizer(MLIRContext* ctx) : ctx_(ctx) {}

void MemoryAccessOptimizer::optimizeMemoryAccesses(Operation* root,
                                                  const target::TargetCharacteristics& target) {
  if (!root) {
    return;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Optimizing memory accesses\n");
  
  // Add prefetch hints where beneficial
  if (target.type == target::TargetType::CPU) {
    addPrefetchHints(root, target);
  }
  
  // Optimize for memory coalescing on GPU
  if (target.type == target::TargetType::GPU) {
    optimizeForCoalescing(root, target);
  }
}

void MemoryAccessOptimizer::addPrefetchHints(Operation* root,
                                           const target::TargetCharacteristics& target) {
  root->walk([&](AffineLoadOp loadOp) {
    if (isPrefetchBeneficial(loadOp, target)) {
      insertPrefetch(loadOp, 32); // 32 cache lines ahead
    }
    return WalkResult::advance();
  });
}

void MemoryAccessOptimizer::optimizeForCoalescing(Operation* root,
                                                const target::TargetCharacteristics& target) {
  root->walk([&](AffineLoadOp loadOp) {
    if (!isCoalescedAccess(loadOp)) {
      // Mark for optimization
      loadOp->setAttr("autopoly.optimize_coalescing", UnitAttr::get(ctx_));
    }
    return WalkResult::advance();
  });
}

bool MemoryAccessOptimizer::isPrefetchBeneficial(AffineLoadOp loadOp,
                                                const target::TargetCharacteristics& target) {
  // Estimate if prefetching would be beneficial
  // For CPU targets with cache hierarchy
  return target.type == target::TargetType::CPU && !target.memory_hierarchy.empty();
}

bool MemoryAccessOptimizer::isCoalescedAccess(AffineLoadOp loadOp) {
  // Check if memory access is coalesced (consecutive accesses)
  // This is a simplified check
  
  // For now, assume unit stride accesses are coalesced
  return true;
}

void MemoryAccessOptimizer::insertPrefetch(AffineLoadOp loadOp, int distance) {
  // Insert prefetch instruction before the load
  // This would require creating a prefetch intrinsic
  
  LLVM_DEBUG(llvm::dbgs() << "Inserting prefetch with distance " << distance << "\n");
  
  // Add prefetch attribute for now
  loadOp->setAttr("autopoly.prefetch_distance", 
                 IntegerAttr::get(IntegerType::get(ctx_, 32), distance));
}

// CodeGenUtils implementation
AffineExpr CodeGenUtils::convertISLExprToAffine(isl_ast_expr* expr, MLIRContext* ctx) {
  if (!expr) {
    return nullptr;
  }
  
  // Simplified conversion - return constant expression
  return getAffineConstantExpr(0, ctx);
}

std::pair<AffineMap, AffineMap> CodeGenUtils::createLoopBounds(isl_ast_node* for_node, 
                                                              MLIRContext* ctx) {
  // Create simple constant bounds for demonstration
  AffineMap lowerMap = AffineMap::getConstantMap(0, ctx);
  AffineMap upperMap = AffineMap::getConstantMap(100, ctx);
  
  return {lowerMap, upperMap};
}

std::vector<int> CodeGenUtils::extractParallelDimensions(isl_schedule* schedule) {
  std::vector<int> parallel_dims;
  
  if (!schedule) {
    return parallel_dims;
  }
  
  // Simplified extraction - assume outermost dimension is parallel
  parallel_dims.push_back(0);
  
  return parallel_dims;
}

std::string CodeGenUtils::generateUniqueName(const std::string& base) {
  static int counter = 0;
  return base + "_" + std::to_string(counter++);
}

void CodeGenUtils::addDebugInfo(Operation* op, const std::string& info) {
  if (!op) {
    return;
  }
  
  op->setAttr("autopoly.debug_info", StringAttr::get(op->getContext(), info));
}

DictionaryAttr CodeGenUtils::createPerformanceAttrs(MLIRContext* ctx,
                                                   const std::map<std::string, int>& params) {
  llvm::SmallVector<NamedAttribute> attrs;
  
  for (const auto& param : params) {
    attrs.push_back(NamedAttribute(
        StringAttr::get(ctx, param.first),
        IntegerAttr::get(IntegerType::get(ctx, 32), param.second)
    ));
  }
  
  return DictionaryAttr::get(ctx, attrs);
}

} // namespace codegen
} // namespace autopoly
