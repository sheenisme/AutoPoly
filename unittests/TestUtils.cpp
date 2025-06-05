//===- TestUtils.cpp - AutoPoly Unit Test Utilities -----------*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements utility classes and functions for AutoPoly unit tests.
//
//===----------------------------------------------------------------------===//

#include "TestUtils.h"

#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "AutoPoly/Target/TargetInfo.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/PassManager.h"

#include <gtest/gtest.h>
#include <memory>
#include <string>

// Forward declarations for ISL
extern "C" {
#include <isl/ctx.h>
}

using namespace mlir;
using namespace mlir::func;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace mlir::memref;

namespace autopoly {
namespace test {

void AutoPolyTestBase::SetUp() {
  mlir_context_ = std::make_unique<MLIRContext>();
  
  // Register all dialects needed for testing
  DialectRegistry registry;
  registry.template insert<FuncDialect>();
  registry.template insert<AffineDialect>();
  registry.template insert<ArithDialect>();
  registry.template insert<MemRefDialect>();
  mlir_context_->appendDialectRegistry(registry);
  mlir_context_->loadAllAvailableDialects();
  
  // Create ISL context
  isl_context_ = isl_ctx_alloc();
}

void AutoPolyTestBase::TearDown() {
  if (isl_context_) {
    isl_ctx_free(isl_context_);
    isl_context_ = nullptr;
  }
  mlir_context_.reset();
}

OwningOpRef<ModuleOp> AutoPolyTestBase::parseMLIR(const std::string& mlir_code) {
  return parseSourceString<ModuleOp>(mlir_code, mlir_context_.get());
}

FuncOp AutoPolyTestBase::createSimpleLoopNest(int num_loops) {
  MLIRContext context;
  OpBuilder builder(&context);
  auto funcType = builder.getFunctionType({}, {});
  auto funcOp = builder.create<FuncOp>(builder.getUnknownLoc(), "simple_loop_nest", funcType);
  Block *entry_block = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);
  Value iv = nullptr;
  for (int i = 0; i < num_loops; ++i) {
    iv = builder.create<AffineForOp>(builder.getUnknownLoc(), 0, 10).getInductionVar();
    builder.setInsertionPointToStart(&iv.getDefiningOp()->getRegion(0).front());
  }
  builder.create<ReturnOp>(builder.getUnknownLoc());
  return funcOp;
}

FuncOp AutoPolyTestBase::createMatMulFunction() {
  MLIRContext context;
  OpBuilder builder(&context);
  
  // Create function type with three memref arguments
  auto memrefType = MemRefType::get({10, 10}, builder.getF32Type());
  auto funcType = builder.getFunctionType(
    {memrefType, memrefType, memrefType}, {});
  
  auto funcOp = builder.create<FuncOp>(
    builder.getUnknownLoc(), "matmul", funcType);
  
  Block* entry_block = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);
  
  auto args = entry_block->getArguments();
  auto A = args[0];
  auto B = args[1];
  auto C = args[2];
  
  // Create loop nest
  auto i = builder.create<AffineForOp>(
    builder.getUnknownLoc(), 0, 10).getInductionVar();
  builder.setInsertionPointToStart(&i.getDefiningOp()->getRegion(0).front());
  
  auto j = builder.create<AffineForOp>(
    builder.getUnknownLoc(), 0, 10).getInductionVar();
  builder.setInsertionPointToStart(&j.getDefiningOp()->getRegion(0).front());
  
  auto k = builder.create<AffineForOp>(
    builder.getUnknownLoc(), 0, 10).getInductionVar();
  builder.setInsertionPointToStart(&k.getDefiningOp()->getRegion(0).front());
  
  // Create memory accesses
  auto loadA = builder.create<AffineLoadOp>(
    builder.getUnknownLoc(), A, ValueRange{i, k});
  auto loadB = builder.create<AffineLoadOp>(
    builder.getUnknownLoc(), B, ValueRange{k, j});
  auto loadC = builder.create<AffineLoadOp>(
    builder.getUnknownLoc(), C, ValueRange{i, j});
  
  // Create multiply and add
  auto mul = builder.create<arith::MulFOp>(
    builder.getUnknownLoc(), loadA, loadB);
  auto add = builder.create<arith::AddFOp>(
    builder.getUnknownLoc(), loadC, mul);
  
  // Store result
  builder.create<AffineStoreOp>(
    builder.getUnknownLoc(), add, C, ValueRange{i, j});
  
  builder.create<ReturnOp>(builder.getUnknownLoc());
  return funcOp;
}

FuncOp AutoPolyTestBase::createComplexAffineFunction() {
  MLIRContext context;
  OpBuilder builder(&context);
  
  auto memrefType = MemRefType::get({10, 10}, builder.getF32Type());
  auto funcType = builder.getFunctionType({memrefType, memrefType}, {});
  
  auto funcOp = builder.create<FuncOp>(
    builder.getUnknownLoc(), "complex_affine", funcType);
  
  Block* entry_block = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);
  
  auto args = entry_block->getArguments();
  auto A = args[0];
  auto B = args[1];
  
  // Create loop nest
  auto i = builder.create<AffineForOp>(
    builder.getUnknownLoc(), 0, 10).getInductionVar();
  builder.setInsertionPointToStart(&i.getDefiningOp()->getRegion(0).front());
  
  auto j = builder.create<AffineForOp>(
    builder.getUnknownLoc(), 0, 10).getInductionVar();
  builder.setInsertionPointToStart(&j.getDefiningOp()->getRegion(0).front());
  
  // Create affine expression: i + j < 15
  auto expr_i = mlir::getAffineDimExpr(0, builder.getContext());
  auto expr_j = mlir::getAffineDimExpr(1, builder.getContext());
  auto expr = expr_i + expr_j;
  auto constraint = mlir::getAffineConstantExpr(15, builder.getContext());
  
  // Create integer set for the condition
  SmallVector<AffineExpr> constraints;
  constraints.push_back(expr - constraint);
  auto set = IntegerSet::get(2, 0, constraints, SmallVector<bool>{true});
  
  auto ifOp = builder.create<AffineIfOp>(
    builder.getUnknownLoc(), set, ValueRange{i, j}, true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  // Create memory access
  auto loadA = builder.create<AffineLoadOp>(
    builder.getUnknownLoc(), A, ValueRange{i, j});
  auto loadB = builder.create<AffineLoadOp>(
    builder.getUnknownLoc(), B, ValueRange{i, j});
  auto mul = builder.create<arith::MulFOp>(
    builder.getUnknownLoc(), loadA, loadB);
  builder.create<AffineStoreOp>(
    builder.getUnknownLoc(), mul, B, ValueRange{i, j});
  builder.create<ReturnOp>(builder.getUnknownLoc());
  return funcOp;
}

// AnalysisTestBase implementation
void AnalysisTestBase::SetUp() {
  AutoPolyTestBase::SetUp();
}

std::unique_ptr<analysis::PolyhedralExtractor> AnalysisTestBase::createExtractor() {
  return std::make_unique<analysis::PolyhedralExtractor>(getISLContext());
}

std::unique_ptr<analysis::DependenceAnalyzer> AnalysisTestBase::createDependenceAnalyzer() {
  return std::make_unique<analysis::DependenceAnalyzer>(getISLContext());
}

// SchedulingTestBase implementation
void SchedulingTestBase::SetUp() {
  AutoPolyTestBase::SetUp();
}

target::TargetCharacteristics SchedulingTestBase::createMockCPUTarget() {
  target::TargetCharacteristics target;
  target.type = target::TargetType::CPU;
  target.name = "MockCPU";
  target.vendor = "Generic";
  target.compute_units = 8;
  target.max_work_group_size = 8;
  target.max_work_item_dimensions = 1;
  target.max_work_item_sizes = {8};
  target::TargetCharacteristics::MemoryInfo l1;
  l1.level = target::MemoryLevel::LOCAL;
  l1.size_bytes = 32 * 1024;
  l1.bandwidth_gb_per_s = 100.0;
  l1.latency_cycles = 1;
  target.memory_hierarchy.push_back(l1);
  target.supports_double_precision = true;
  target.supports_atomic_operations = true;
  target.supports_vectorization = true;
  target.supports_local_memory = true;
  target.peak_compute_throughput = 100.0;
  target.memory_coalescing_factor = 1.0;
  return target;
}

target::TargetCharacteristics SchedulingTestBase::createMockGPUTarget() {
  target::TargetCharacteristics target;
  target.type = target::TargetType::GPU;
  target.name = "MockGPU";
  target.compute_units = 2048;
  target.supports_vectorization = true;
  target.max_work_group_size = 1024;
  target.max_work_item_dimensions = 3;
  target.supports_local_memory = true;
  
  // Add memory hierarchy
  target::TargetCharacteristics::MemoryInfo shared;
  shared.level = target::MemoryLevel::SHARED;
  shared.size_bytes = 48 * 1024;
  shared.latency_cycles = 1;
  shared.bandwidth_gb_per_s = 1000.0;
  target.memory_hierarchy.push_back(shared);
  
  target::TargetCharacteristics::MemoryInfo global;
  global.level = target::MemoryLevel::GLOBAL;
  global.size_bytes = 8ULL * 1024 * 1024 * 1024; // 8GB
  global.latency_cycles = 400;
  global.bandwidth_gb_per_s = 500.0;
  target.memory_hierarchy.push_back(global);
  
  return target;
}

target::TargetCharacteristics SchedulingTestBase::createMockFPGATarget() {
  target::TargetCharacteristics target;
  target.type = target::TargetType::FPGA;
  target.name = "MockFPGA";
  target.compute_units = 256;
  target.supports_vectorization = false;
  target.max_work_group_size = 1;
  target.max_work_item_dimensions = 1;
  target.supports_local_memory = true;
  
  return target;
}

target::TargetCharacteristics SchedulingTestBase::createMockNPUTarget() {
  target::TargetCharacteristics target;
  target.type = target::TargetType::NPU;
  target.name = "MockNPU";
  target.compute_units = 64;
  target.supports_vectorization = true;
  target.max_work_group_size = 1;
  target.max_work_item_dimensions = 1;
  target.supports_local_memory = true;
  
  return target;
}

// TransformTestBase implementation
void TransformTestBase::SetUp() {
  AutoPolyTestBase::SetUp();
}

std::unique_ptr<analysis::PolyhedralModel> TransformTestBase::createSampleModel() {
  auto funcOp = createMatMulFunction();
  auto extractor = std::make_unique<analysis::PolyhedralExtractor>(getISLContext());
  return extractor->extractFromFunction(funcOp);
}

std::unique_ptr<analysis::DependenceInfo> TransformTestBase::createSampleDependences() {
  auto model = createSampleModel();
  if (!model) return nullptr;
  
  auto analyzer = std::make_unique<analysis::DependenceAnalyzer>(getISLContext());
  return analyzer->analyze(*model);
}

// CodeGenTestBase implementation
void CodeGenTestBase::SetUp() {
  AutoPolyTestBase::SetUp();
}

isl_schedule* CodeGenTestBase::createSampleSchedule() {
  // Create simple schedule for testing
  // This is a simplified implementation
  return nullptr;
}

bool CodeGenTestBase::verifyGeneratedOps(Operation* op) {
  if (!op) return false;
  
  // Basic verification - check that operation is valid
  return op->isRegistered();
}

// PassTestBase implementation
void PassTestBase::SetUp() {
  AutoPolyTestBase::SetUp();
}

LogicalResult PassTestBase::runPassOnFunction(Pass* pass, FuncOp funcOp) {
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    return failure();
  }
  mlir::PassManager pm(moduleOp.getContext());
  pm.addPass(std::unique_ptr<mlir::Pass>(pass));
  return pm.run(moduleOp);
}

bool PassTestBase::wasModified(FuncOp before, FuncOp after) {
  // Simple comparison based on operation count
  int before_count = 0, after_count = 0;
  
  before.walk([&](Operation*) { before_count++; });
  after.walk([&](Operation*) { after_count++; });
  
  return before_count != after_count;
}

// Utility functions implementation
bool compareMLIROps(Operation* op1, Operation* op2) {
  if (!op1 || !op2) return false;
  return op1->getName() == op2->getName();
}

bool hasAttribute(Operation* op, const std::string& attr_name) {
  if (!op) return false;
  return op->hasAttr(attr_name);
}

int countLoops(FuncOp funcOp) {
  int count = 0;
  funcOp.walk([&](AffineForOp) { count++; });
  return count;
}

int countParallelLoops(FuncOp funcOp) {
  int count = 0;
  funcOp.walk([&](AffineParallelOp) { count++; });
  return count;
}

std::vector<int> extractTileSizes(FuncOp funcOp) {
  std::vector<int> tile_sizes;
  // Simplified implementation - would extract from attributes
  return tile_sizes;
}

bool isParallelized(FuncOp funcOp) {
  return countParallelLoops(funcOp) > 0;
}

bool isFused(FuncOp funcOp) {
  // Simplified check - look for fusion attributes
  bool has_fusion = false;
  funcOp.walk([&](Operation* op) {
    if (hasAttribute(op, "autopoly.fused")) {
      has_fusion = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return has_fusion;
}

// MockTargetDetector implementation
MockTargetDetector::MockTargetDetector() = default;

std::vector<target::TargetCharacteristics> MockTargetDetector::detectTargets() {
  return mock_targets_;
}

bool MockTargetDetector::isTargetAvailable(target::TargetType type) {
  for (const auto& target : mock_targets_) {
    if (target.type == type) return true;
  }
  return false;
}

target::TargetCharacteristics MockTargetDetector::getDefaultTarget() {
  if (!mock_targets_.empty()) {
    return mock_targets_[0];
  }
  // Return mock CPU target as default
  target::TargetCharacteristics default_target;
  default_target.type = target::TargetType::CPU;
  default_target.name = "DefaultMockCPU";
  default_target.compute_units = 4;
  return default_target;
}

void MockTargetDetector::addMockTarget(const target::TargetCharacteristics& target) {
  mock_targets_.push_back(target);
}

void MockTargetDetector::clearMockTargets() {
  mock_targets_.clear();
}

// MLIRTestCases implementation
std::string MLIRTestCases::getSimpleLoopNest() {
  return R"(
    func.func @simple_loop() {
      affine.for %i = 0 to 10 {
        affine.for %j = 0 to 10 {
          %c1 = arith.constant 1 : index
        }
      }
      return
    }
  )";
}

std::string MLIRTestCases::getMatrixMultiplication() {
  return R"(
    func.func @matmul(%A: memref<100x100xf32>, %B: memref<100x100xf32>, %C: memref<100x100xf32>) {
      affine.for %i = 0 to 100 {
        affine.for %j = 0 to 100 {
          affine.for %k = 0 to 100 {
            %a = affine.load %A[%i, %k] : memref<100x100xf32>
            %b = affine.load %B[%k, %j] : memref<100x100xf32>
            %c = affine.load %C[%i, %j] : memref<100x100xf32>
            %prod = arith.mulf %a, %b : f32
            %sum = arith.addf %c, %prod : f32
            affine.store %sum, %C[%i, %j] : memref<100x100xf32>
          }
        }
      }
      return
    }
  )";
}

std::string MLIRTestCases::getLoopWithConditional() {
  return R"(
    func.func @loop_with_if(%A: memref<100x100xf32>, %B: memref<100x100xf32>) {
      affine.for %i = 0 to 100 {
        affine.for %j = 0 to 100 {
          affine.if affine_set<(d0, d1) : (d0 + d1 - 50 < 0)>(%i, %j) {
            %a = affine.load %A[%i, %j] : memref<100x100xf32>
            %c2 = arith.constant 2.0 : f32
            %result = arith.mulf %a, %c2 : f32
            affine.store %result, %B[%i, %j] : memref<100x100xf32>
          }
        }
      }
      return
    }
  )";
}

std::string MLIRTestCases::getNestedLoopsWithBounds() {
  return R"(
    func.func @nested_bounds(%N: index, %M: index, %A: memref<?x?xf32>) {
      affine.for %i = 0 to %N {
        affine.for %j = 0 to %M {
          %c1 = arith.constant 1.0 : f32
          affine.store %c1, %A[%i, %j] : memref<?x?xf32>
        }
      }
      return
    }
  )";
}

std::string MLIRTestCases::getFunctionWithReturn() {
  return R"(
    func.func @func_with_return(%A: memref<10xf32>) -> f32 {
      %sum = arith.constant 0.0 : f32
      %result = affine.for %i = 0 to 10 iter_args(%iter = %sum) -> f32 {
        %val = affine.load %A[%i] : memref<10xf32>
        %new_sum = arith.addf %iter, %val : f32
        affine.yield %new_sum : f32
      }
      return %result : f32
    }
  )";
}

std::string MLIRTestCases::getFunctionWithYield() {
  return R"(
    func.func @func_with_yield(%A: memref<10xf32>) -> f32 {
      %init = arith.constant 0.0 : f32
      %result = affine.for %i = 0 to 10 iter_args(%acc = %init) -> f32 {
        %val = affine.load %A[%i] : memref<10xf32>
        %sum = arith.addf %acc, %val : f32
        affine.yield %sum : f32
      }
      return %result : f32
    }
  )";
}

// Add remaining implementations...
std::string MLIRTestCases::getLoopWithMemoryAccesses() { return getMatrixMultiplication(); }
std::string MLIRTestCases::getComplexAffineExpressions() { return getLoopWithConditional(); }
std::string MLIRTestCases::getMultipleStatementGroups() { return getMatrixMultiplication(); }
std::string MLIRTestCases::getFunctionWithParameters() { return getNestedLoopsWithBounds(); }
std::string MLIRTestCases::getFunctionWithIf() { return getLoopWithConditional(); }

// PerformanceTestUtils implementation
double PerformanceTestUtils::measureExecutionTime(std::function<void()> func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return duration.count() / 1000.0; // Return milliseconds
}

double PerformanceTestUtils::comparePerformance(std::function<void()> baseline,
                                               std::function<void()> optimized) {
  double baseline_time = measureExecutionTime(baseline);
  double optimized_time = measureExecutionTime(optimized);
  
  return baseline_time / optimized_time; // Speedup ratio
}

std::string PerformanceTestUtils::generatePerformanceReport(
    const std::map<std::string, double>& results) {
  std::string report = "Performance Test Results:\n";
  report += "=========================\n";
  
  for (const auto& result : results) {
    report += result.first + ": " + std::to_string(result.second) + "ms\n";
  }
  
  return report;
}

} // namespace test
} // namespace autopoly
