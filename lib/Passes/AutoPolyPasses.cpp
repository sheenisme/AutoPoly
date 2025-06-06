//===- AutoPolyPasses.cpp - AutoPoly Passes Implementation ----------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file implements MLIR passes for the AutoPoly polyhedral scheduling
// framework, providing automatic scheduling transformations for affine
// dialect operations.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Passes/AutoPolyPasses.h"
#include "AutoPoly/Analysis/DependenceAnalysis.h"
#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "AutoPoly/CodeGen/AffineCodeGeneration.h"
#include "AutoPoly/Scheduling/SchedulingStrategy.h"
#include "AutoPoly/Target/TargetInfo.h"
#include "AutoPoly/Transform/SchedulingTransform.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/Debug.h"

// ISL headers
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/ast_type.h>
#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/schedule.h>
#include <isl/space.h>
#include <isl/schedule_node.h>
#include <isl/constraint.h>
#include <isl/options.h>

#include <chrono>

#define DEBUG_TYPE "autopoly-passes"

using namespace mlir;

namespace autopoly {
namespace passes {

// Global statistics instance
AutoPolyPassStatistics g_pass_statistics;

void AutoPolyPassStatistics::reset() {
  functions_analyzed = 0;
  affine_loops_found = 0;
  polyhedral_models_extracted = 0;
  dependence_relations_computed = 0;
  functions_transformed = 0;
  loops_tiled = 0;
  loops_parallelized = 0;
  loops_fused = 0;
  loops_skewed = 0;
  total_analysis_time = 0.0;
  total_transformation_time = 0.0;
  estimated_total_speedup = 1.0;
  detected_target_type.clear();
  available_targets = 0;
  analysis_failures = 0;
  transformation_failures = 0;
}

std::string AutoPolyPassStatistics::toString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  
  os << "AutoPoly Pass Statistics:\n";
  os << "========================\n";
  os << "Analysis:\n";
  os << "  Functions analyzed: " << functions_analyzed << "\n";
  os << "  Affine loops found: " << affine_loops_found << "\n";
  os << "  Polyhedral models extracted: " << polyhedral_models_extracted << "\n";
  os << "  Dependence relations computed: " << dependence_relations_computed << "\n";
  os << "  Analysis failures: " << analysis_failures << "\n";
  
  os << "\nTransformation:\n";
  os << "  Functions transformed: " << functions_transformed << "\n";
  os << "  Loops tiled: " << loops_tiled << "\n";
  os << "  Loops parallelized: " << loops_parallelized << "\n";
  os << "  Loops fused: " << loops_fused << "\n";
  os << "  Loops skewed: " << loops_skewed << "\n";
  os << "  Transformation failures: " << transformation_failures << "\n";
  
  os << "\nPerformance:\n";
  os << "  Total analysis time: " << total_analysis_time << " seconds\n";
  os << "  Total transformation time: " << total_transformation_time << " seconds\n";
  os << "  Estimated total speedup: " << estimated_total_speedup << "x\n";
  
  os << "\nTarget Information:\n";
  os << "  Detected target type: " << detected_target_type << "\n";
  os << "  Available targets: " << available_targets << "\n";
  
  return os.str();
}

// AutoPolySchedulingPass implementation
AutoPolySchedulingPass::AutoPolySchedulingPass() : mlir::OperationPass<mlir::func::FuncOp>(mlir::TypeID::get<AutoPolySchedulingPass>()) {}

AutoPolySchedulingPass::AutoPolySchedulingPass(const AutoPolyPassOptions& options)
    : mlir::OperationPass<mlir::func::FuncOp>(mlir::TypeID::get<AutoPolySchedulingPass>()), options_(options) {}

AutoPolySchedulingPass::AutoPolySchedulingPass(const AutoPolySchedulingPass& other)
    : mlir::OperationPass<mlir::func::FuncOp>(mlir::TypeID::get<AutoPolySchedulingPass>()), options_(other.options_) {}

std::unique_ptr<mlir::Pass> AutoPolySchedulingPass::clonePass() const {
  return std::make_unique<AutoPolySchedulingPass>(*this);
}

void AutoPolySchedulingPass::getDependentDialects(mlir::DialectRegistry& registry) const {
  registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
}

void AutoPolySchedulingPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  
  LLVM_DEBUG(llvm::dbgs() << "Running AutoPoly scheduling pass on function: " 
                          << funcOp.getName() << "\n");
  
  g_pass_statistics.functions_analyzed++;
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Check if function is suitable for polyhedral analysis
  if (!analyzeFunction(funcOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Function not suitable for polyhedral optimization\n");
    g_pass_statistics.analysis_failures++;
    return;
  }
  
  // Perform transformation
  if (!transformFunction(funcOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to transform function\n");
    g_pass_statistics.transformation_failures++;
    return;
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  g_pass_statistics.total_transformation_time += duration.count() / 1000.0;
  
  g_pass_statistics.functions_transformed++;
  
  if (options_.debug_mode) {
    generateStatistics();
  }
}

bool AutoPolySchedulingPass::analyzeFunction(func::FuncOp funcOp) {
  // Check for affine loops
  bool hasAffineLoops = false;
  int loopCount = 0;
  
  funcOp.walk([&](mlir::affine::AffineForOp forOp) {
    hasAffineLoops = true;
    loopCount++;
  });
  
  g_pass_statistics.affine_loops_found += loopCount;
  
  if (!hasAffineLoops) {
    LLVM_DEBUG(llvm::dbgs() << "No affine loops found in function\n");
    return false;
  }
  
  // Check if all regions are affine
  bool allAffine = true;
  funcOp.walk([&](Region* region) {
    if (!analysis::PolyhedralUtils::isRegionAffine(*region)) {
      allAffine = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  if (!allAffine) {
    LLVM_DEBUG(llvm::dbgs() << "Function contains non-affine operations\n");
    return false;
  }
  
  return true;
}

bool AutoPolySchedulingPass::transformFunction(func::FuncOp funcOp) {
  // Create ISL context
  isl_ctx* ctx = analysis::PolyhedralUtils::createContext();
  
  // Extract polyhedral model
  analysis::PolyhedralExtractor extractor(ctx);
  auto model = extractor.extractFromFunction(funcOp);
  
  if (!model) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to extract polyhedral model\n");
    isl_ctx_free(ctx);
    return false;
  }
  
  g_pass_statistics.polyhedral_models_extracted++;
  
  if (options_.dump_polyhedral_model) {
    llvm::outs() << "Extracted polyhedral model for function: " << funcOp.getName() << "\n";
    llvm::outs() << "Number of statements: " << model->getStatements().size() << "\n";
    llvm::outs() << "Number of arrays: " << model->getArrayAccesses().size() << "\n";
  }
  
  // Perform dependence analysis
  analysis::DependenceAnalyzer depAnalyzer(ctx);
  auto dependences = depAnalyzer.analyze(*model);
  
  if (!dependences) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to perform dependence analysis\n");
    isl_ctx_free(ctx);
    return false;
  }
  
  g_pass_statistics.dependence_relations_computed += dependences->getDependences().size();
  
  if (options_.dump_dependences) {
    llvm::outs() << "Dependence analysis results:\n";
    llvm::outs() << "Number of dependences: " << dependences->getDependences().size() << "\n";
  }
  
  // Detect target
  auto detector = target::TargetDetectorFactory::createDetector();
  target::TargetCharacteristics target;
  
  if (options_.target_type == "auto") {
    target = detector->getDefaultTarget();
  } else {
    auto target_type = target::TargetUtils::stringToTargetType(options_.target_type);
    if (detector->isTargetAvailable(target_type)) {
      // Get first available target of this type
      auto targets = detector->detectTargets();
      for (const auto& t : targets) {
        if (t.type == target_type) {
          target = t;
          break;
        }
      }
    } else {
      target = detector->getDefaultTarget();
      LLVM_DEBUG(llvm::dbgs() << "Requested target type not available, using default\n");
    }
  }
  
  g_pass_statistics.detected_target_type = target::TargetUtils::targetTypeToString(target.type);
  
  // Select scheduling strategy
  scheduling::SchedulingStrategyManager strategyManager;
  auto strategy = strategyManager.selectStrategy(target);
  
  if (!strategy) {
    LLVM_DEBUG(llvm::dbgs() << "No suitable scheduling strategy found\n");
    isl_ctx_free(ctx);
    return false;
  }
  
  // Get scheduling parameters
  auto params = strategy->getParameters(target);
  
  // Override with user-specified parameters
  if (!options_.tile_sizes.empty()) {
    params.tile_sizes = options_.tile_sizes;
  }
  
  params.enable_tiling = options_.enable_tiling;
  params.enable_loop_fusion = options_.enable_fusion;
  params.enable_nested_parallelism = options_.enable_parallelization;
  params.enable_skewing = options_.enable_skewing;
  params.max_parallel_depth = options_.max_parallel_depth;
  
  // Perform scheduling transformation
  transform::SchedulingTransformer transformer(ctx);
  auto result = transformer.transform(*model, *dependences, target, *strategy);
  
  if (!result.transformation_successful) {
    LLVM_DEBUG(llvm::dbgs() << "Scheduling transformation failed: " 
                            << result.error_message << "\n");
    isl_ctx_free(ctx);
    return false;
  }
  
  if (options_.dump_schedules) {
    llvm::outs() << "Scheduling transformation results:\n";
    llvm::outs() << "Applied techniques: ";
    for (auto technique : result.applied_techniques) {
      llvm::outs() << scheduling::SchedulingUtils::techniqueToString(technique) << " ";
    }
    llvm::outs() << "\n";
    llvm::outs() << "Estimated speedup: " << result.estimated_speedup << "x\n";
  }
  
  // Update statistics
  g_pass_statistics.estimated_total_speedup *= result.estimated_speedup;
  
  for (auto technique : result.applied_techniques) {
    switch (technique) {
      case scheduling::OptimizationTechnique::TILING:
        g_pass_statistics.loops_tiled++;
        break;
      case scheduling::OptimizationTechnique::PARALLELIZATION:
        g_pass_statistics.loops_parallelized++;
        break;
      case scheduling::OptimizationTechnique::FUSION:
        g_pass_statistics.loops_fused++;
        break;
      case scheduling::OptimizationTechnique::SKEWING:
        g_pass_statistics.loops_skewed++;
        break;
      default:
        break;
    }
  }
  
  // Generate optimized code
  MLIRContext* mlirCtx = &getContext();
  OpBuilder builder(mlirCtx);
  
  codegen::AffineCodeGenerator codeGen(mlirCtx, ctx);
  codegen::CodeGenOptions codeGenOptions;
  codeGenOptions.generate_parallel_loops = options_.enable_parallelization;
  codeGenOptions.generate_vector_hints = options_.enable_vectorization;
  codeGenOptions.verify_generated_code = options_.verify_transformations;
  
  auto codeGenResult = codeGen.generateCodeWithOptions(
      result.transformed_schedule, *model, result, builder, funcOp.getLoc(), codeGenOptions);
  
  if (!codeGenResult.generation_successful) {
    LLVM_DEBUG(llvm::dbgs() << "Code generation failed: " 
                            << codeGenResult.error_message << "\n");
    isl_ctx_free(ctx);
    return false;
  }
  
  // Replace original code with optimized version
  if (!codeGen.replaceOriginalCode(funcOp.getOperation(), codeGenResult)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to replace original code\n");
    isl_ctx_free(ctx);
    return false;
  }
  
  // Verify transformations if requested
  if (options_.verify_transformations) {
    if (!codegen::CodeVerifier::verifyAffineCode(funcOp.getOperation())) {
      LLVM_DEBUG(llvm::dbgs() << "Verification failed after transformation\n");
      isl_ctx_free(ctx);
      return false;
    }
  }
  
  isl_ctx_free(ctx);
  return true;
}

void AutoPolySchedulingPass::generateStatistics() {
  if (options_.debug_mode) {
    llvm::outs() << g_pass_statistics.toString();
  }
}

// Factory functions
std::unique_ptr<Pass> createAutoPolySchedulingPass() {
  return std::make_unique<AutoPolySchedulingPass>();
}

std::unique_ptr<Pass> createAutoPolySchedulingPass(const AutoPolyPassOptions& options) {
  return std::make_unique<AutoPolySchedulingPass>(options);
}

// PolyhedralAnalysisPass implementation
PolyhedralAnalysisPass::PolyhedralAnalysisPass() 
    : mlir::OperationPass<mlir::func::FuncOp>(mlir::TypeID::get<PolyhedralAnalysisPass>()) {}

std::unique_ptr<mlir::Pass> PolyhedralAnalysisPass::clonePass() const {
  return std::make_unique<PolyhedralAnalysisPass>();
}

void PolyhedralAnalysisPass::getDependentDialects(mlir::DialectRegistry& registry) const {
  registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect>();
}

void PolyhedralAnalysisPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  
  LLVM_DEBUG(llvm::dbgs() << "Running polyhedral analysis pass on function: " 
                          << funcOp.getName() << "\n");
  
  analyzeAffineRegions(funcOp);
  
  if (enable_debug_output_) {
    dumpPolyhedralModel(funcOp);
  }
}

void PolyhedralAnalysisPass::analyzeAffineRegions(func::FuncOp funcOp) {
  isl_ctx* ctx = analysis::PolyhedralUtils::createContext();
  analysis::PolyhedralExtractor extractor(ctx);
  
  auto model = extractor.extractFromFunction(funcOp);
  
  if (model) {
    LLVM_DEBUG(llvm::dbgs() << "Successfully extracted polyhedral model\n");
    LLVM_DEBUG(llvm::dbgs() << "Number of statements: " << model->getStatements().size() << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Failed to extract polyhedral model\n");
  }
  
  isl_ctx_free(ctx);
}

void PolyhedralAnalysisPass::dumpPolyhedralModel(func::FuncOp funcOp) {
  llvm::outs() << "Polyhedral Analysis Results for function: " << funcOp.getName() << "\n";
  llvm::outs() << "==========================================\n";
  // Additional detailed output would go here
}

// DependenceAnalysisPass implementation
DependenceAnalysisPass::DependenceAnalysisPass() 
    : mlir::OperationPass<mlir::func::FuncOp>(mlir::TypeID::get<DependenceAnalysisPass>()) {}

std::unique_ptr<mlir::Pass> DependenceAnalysisPass::clonePass() const {
  return std::make_unique<DependenceAnalysisPass>();
}

void DependenceAnalysisPass::getDependentDialects(mlir::DialectRegistry& registry) const {
  registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect>();
}

void DependenceAnalysisPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  
  LLVM_DEBUG(llvm::dbgs() << "Running dependence analysis pass on function: " 
                          << funcOp.getName() << "\n");
  
  analyzeDependences(funcOp);
  
  if (enable_debug_output_) {
    dumpDependenceInformation(funcOp);
  }
}

void DependenceAnalysisPass::analyzeDependences(func::FuncOp funcOp) {
  isl_ctx* ctx = analysis::PolyhedralUtils::createContext();
  
  // Extract polyhedral model first
  analysis::PolyhedralExtractor extractor(ctx);
  auto model = extractor.extractFromFunction(funcOp);
  
  if (!model) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot perform dependence analysis without polyhedral model\n");
    isl_ctx_free(ctx);
    return;
  }
  
  // Analyze dependences
  analysis::DependenceAnalyzer analyzer(ctx);
  auto dependences = analyzer.analyze(*model);
  
  if (dependences) {
    LLVM_DEBUG(llvm::dbgs() << "Successfully analyzed dependences\n");
    LLVM_DEBUG(llvm::dbgs() << "Number of dependences: " << dependences->getDependences().size() << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Failed to analyze dependences\n");
  }
  
  isl_ctx_free(ctx);
}

void DependenceAnalysisPass::dumpDependenceInformation(func::FuncOp funcOp) {
  llvm::outs() << "Dependence Analysis Results for function: " << funcOp.getName() << "\n";
  llvm::outs() << "=========================================\n";
  // Additional detailed output would go here
}

// TargetDetectionPass implementation
TargetDetectionPass::TargetDetectionPass() 
    : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<TargetDetectionPass>()) {}

std::unique_ptr<mlir::Pass> TargetDetectionPass::clonePass() const {
  return std::make_unique<TargetDetectionPass>();
}

void TargetDetectionPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  
  LLVM_DEBUG(llvm::dbgs() << "Running target detection pass on module\n");
  
  detectTargets(moduleOp);
  reportTargetInformation();
}

void TargetDetectionPass::detectTargets(ModuleOp moduleOp) {
  auto detector = target::TargetDetectorFactory::createDetector();
  auto targets = detector->detectTargets();
  
  g_pass_statistics.available_targets = targets.size();
  
  if (!targets.empty()) {
    g_pass_statistics.detected_target_type = 
        target::TargetUtils::targetTypeToString(targets[0].type);
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Detected " << targets.size() << " available targets\n");
}

void TargetDetectionPass::reportTargetInformation() {
  llvm::outs() << "Target Detection Results:\n";
  llvm::outs() << "========================\n";
  llvm::outs() << "Available targets: " << g_pass_statistics.available_targets << "\n";
  llvm::outs() << "Primary target type: " << g_pass_statistics.detected_target_type << "\n";
}

// AutoPolyPipelineBuilder implementation
void AutoPolyPipelineBuilder::addAutoPolyPasses(PassManager& pm, const AutoPolyPassOptions& options) {
  // Add analysis passes first
  addAnalysisPasses(pm);
  
  // Add transformation passes
  addTransformationPasses(pm, options);
}

void AutoPolyPipelineBuilder::addAnalysisPasses(PassManager& pm) {
  pm.addPass(createTargetDetectionPass());
  pm.addPass(createPolyhedralAnalysisPass());
  pm.addPass(createDependenceAnalysisPass());
}

void AutoPolyPipelineBuilder::addTransformationPasses(PassManager& pm, const AutoPolyPassOptions& options) {
  pm.addPass(createAutoPolySchedulingPass(options));
}

std::unique_ptr<PassManager> AutoPolyPipelineBuilder::createPipeline(MLIRContext* context,
                                                                    const AutoPolyPassOptions& options) {
  auto pm = std::make_unique<PassManager>(context);
  addAutoPolyPasses(*pm, options);
  return pm;
}

// Factory functions
std::unique_ptr<Pass> createPolyhedralAnalysisPass() {
  return std::make_unique<PolyhedralAnalysisPass>();
}

std::unique_ptr<Pass> createDependenceAnalysisPass() {
  return std::make_unique<DependenceAnalysisPass>();
}

std::unique_ptr<Pass> createTargetDetectionPass() {
  return std::make_unique<TargetDetectionPass>();
}

// Pass registration
void registerAutoPolyPasses() {
  mlir::PassRegistration<AutoPolySchedulingPass> autoPolyPass(
      []() { return std::make_unique<AutoPolySchedulingPass>(); });
  
  mlir::PassRegistration<PolyhedralAnalysisPass> polyAnalysisPass(
      []() { return std::make_unique<PolyhedralAnalysisPass>(); });
  
  mlir::PassRegistration<DependenceAnalysisPass> depAnalysisPass(
      []() { return std::make_unique<DependenceAnalysisPass>(); });
  
  mlir::PassRegistration<TargetDetectionPass> targetDetectionPass(
      []() { return std::make_unique<TargetDetectionPass>(); });
}

} // namespace passes
} // namespace autopoly
