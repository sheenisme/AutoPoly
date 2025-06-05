//===- TestUtils.cpp - AutoPoly Unit Test Utilities -----------*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements utility classes and functions for AutoPoly unit tests.
//
//===----------------------------------------------------------------------===//

#include "TestUtils.h"
#include "AutoPoly/Analysis/PolyhedralExtraction.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/InitAllDialects.h"

#include <isl/ctx.h>
#include <chrono>

using namespace mlir;

namespace autopoly {
namespace test {

// AutoPolyTestBase implementation
void AutoPolyTestBase::SetUp() {
  mlir_context_ = std::make_unique<MLIRContext>();
  
  // Register all dialects needed for testing
  DialectRegistry registry;
  registerAllDialects(registry);
  mlir_context_->appendDialectRegistry(registry);
  mlir_context_->loadAllAvailableDialects();
  
  // Create ISL context
  isl_context_ = analysis::PolyhedralUtils::createContext();
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

func::FuncOp AutoPolyTestBase::createSimpleLoopNest(int num_loops) {
  OpBuilder builder(mlir_context_.get());
  auto module = ModuleOp::create(builder.getUnknownLoc());
  
  auto func_type = builder.getFunctionType({}, {});
  auto func_op = func::FuncOp::create(builder.getUnknownLoc(), "test_func", func_type);
  
  auto entry_block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);
  
  // Create nested loops
  affine::AffineForOp current_loop = nullptr;
  for (int i = 0; i < num_loops; ++i) {
    auto loop = builder.create<affine::AffineForOp>(
        builder.getUnknownLoc(), 0, 10, 1);
    
    if (current_loop) {
      builder.setInsertionPointToStart(current_loop.getBody());
    }
    current_loop = loop;
    builder.setInsertionPointToStart(loop.getBody());
  }
  
  // Add simple computation in innermost loop
  auto const_op = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
  
  builder.setInsertionPointToEnd(entry_block);
  builder.create<func::ReturnOp>(builder.getUnknownLoc());
  
  module.push_back(func_op);
  return func_op;
}

func::FuncOp AutoPolyTestBase::createMatMulFunction() {
  OpBuilder builder(mlir_context_.get());
  auto module = ModuleOp::create(builder.getUnknownLoc());
  
  // Create memref types
  auto memref_type = MemRefType::get({100, 100}, builder.getF32Type());
  
  auto func_type = builder.getFunctionType(
      {memref_type, memref_type, memref_type}, {});
  auto func_op = func::FuncOp::create(builder.getUnknownLoc(), "matmul", func_type);
  
  auto entry_block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);
  
  auto args = entry_block.getArguments();
  Value A = args[0], B = args[1], C = args[2];
  
  // Create triple nested loop for matrix multiplication
  auto i_loop = builder.create<affine::AffineForOp>(builder.getUnknownLoc(), 0, 100, 1);
  builder.setInsertionPointToStart(i_loop.getBody());
  
  auto j_loop = builder.create<affine::AffineForOp>(builder.getUnknownLoc(), 0, 100, 1);
  builder.setInsertionPointToStart(j_loop.getBody());
  
  auto k_loop = builder.create<affine::AffineForOp>(builder.getUnknownLoc(), 0, 100, 1);
  builder.setInsertionPointToStart(k_loop.getBody());
  
  // C[i][j] += A[i][k] * B[k][j]
  Value i = i_loop.getInductionVar();
  Value j = j_loop.getInductionVar();
  Value k = k_loop.getInductionVar();
  
  auto a_load = builder.create<affine::AffineLoadOp>(
      builder.getUnknownLoc(), A, ValueRange{i, k});
  auto b_load = builder.create<affine::AffineLoadOp>(
      builder.getUnknownLoc(), B, ValueRange{k, j});
  auto c_load = builder.create<affine::AffineLoadOp>(
      builder.getUnknownLoc(), C, ValueRange{i, j});
  
  auto mul = builder.create<arith::MulFOp>(
      builder.getUnknownLoc(), a_load, b_load);
  auto add = builder.create<arith::AddFOp>(
      builder.getUnknownLoc(), c_load, mul);
  
  builder.create<affine::AffineStoreOp>(
      builder.getUnknownLoc(), add, C, ValueRange{i, j});
  
  builder.setInsertionPointToEnd(entry_block);
  builder.create<func::ReturnOp>(builder.getUnknownLoc());
  
  module.push_back(func_op);
  return func_op;
}

func::FuncOp AutoPolyTestBase::createComplexAffineFunction() {
  OpBuilder builder(mlir_context_.get());
  auto module = ModuleOp::create(builder.getUnknownLoc());
  
  auto memref_type = MemRefType::get({100, 100}, builder.getF32Type());
  auto func_type = builder.getFunctionType({memref_type, memref_type}, {});
  auto func_op = func::FuncOp::create(builder.getUnknownLoc(), "complex_func", func_type);
  
  auto entry_block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);
  
  auto args = entry_block.getArguments();
  Value A = args[0], B = args[1];
  
  // Create loop with affine.if
  auto i_loop = builder.create<affine::AffineForOp>(builder.getUnknownLoc(), 0, 100, 1);
  builder.setInsertionPointToStart(i_loop.getBody());
  
  auto j_loop = builder.create<affine::AffineForOp>(builder.getUnknownLoc(), 0, 100, 1);
  builder.setInsertionPointToStart(j_loop.getBody());
  
  Value i = i_loop.getInductionVar();
  Value j = j_loop.getInductionVar();
  
  // Create affine.if: if i + j < 50
  auto constraint = builder.getAffineConstantExpr(50);
  auto expr = builder.getAffineDimExpr(0) + builder.getAffineDimExpr(1);
  auto condition = IntegerSet::get(2, 0, {expr - constraint}, {false});
  
  auto if_op = builder.create<affine::AffineIfOp>(
      builder.getUnknownLoc(), condition, ValueRange{i, j}, false);
  
  // Then block
  builder.setInsertionPointToStart(if_op.getThenBlock());
  auto a_load = builder.create<affine::AffineLoadOp>(
      builder.getUnknownLoc(), A, ValueRange{i, j});
  auto const_val = builder.create<arith::ConstantFloatOp>(
      builder.getUnknownLoc(), llvm::APFloat(2.0f), builder.getF32Type());
  auto mul = builder.create<arith::MulFOp>(
      builder.getUnknownLoc(), a_load, const_val);
  builder.create<affine::AffineStoreOp>(
      builder.getUnknownLoc(), mul, B, ValueRange{i, j});
  
  builder.setInsertionPointToEnd(entry_block);
  builder.create<func::ReturnOp>(builder.getUnknownLoc());
  
  module.push_back(func_op);
  return func_op;
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
  target.compute_units = 8;
  target.supports_vectorization = true;
  target.max_work_group_size = 1;
  target.max_work_item_dimensions = 1;
  target.supports_local_memory = false;
  
  // Add memory hierarchy
  target::MemoryLevel l1;
  l1.level = target::MemoryLevel::LOCAL;
  l1.size_bytes = 32 * 1024;
  l1.latency_cycles = 1;
  l1.bandwidth_gb_per_s = 100.0;
  target.memory_hierarchy.push_back(l1);
  
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
  target::MemoryLevel shared;
  shared.level = target::MemoryLevel::SHARED;
  shared.size_bytes = 48 * 1024;
  shared.latency_cycles = 1;
  shared.bandwidth_gb_per_s = 1000.0;
  target.memory_hierarchy.push_back(shared);
  
  target::MemoryLevel global;
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

LogicalResult PassTestBase::runPassOnFunction(Pass* pass, func::FuncOp funcOp) {
  // This is a simplified implementation
  // In practice would create proper pass manager
  return success();
}

bool PassTestBase::wasModified(func::FuncOp before, func::FuncOp after) {
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

int countLoops(func::FuncOp funcOp) {
  int count = 0;
  funcOp.walk([&](affine::AffineForOp) { count++; });
  return count;
}

int countParallelLoops(func::FuncOp funcOp) {
  int count = 0;
  funcOp.walk([&](affine::AffineParallelOp) { count++; });
  return count;
}

std::vector<int> extractTileSizes(func::FuncOp funcOp) {
  std::vector<int> tile_sizes;
  // Simplified implementation - would extract from attributes
  return tile_sizes;
}

bool isParallelized(func::FuncOp funcOp) {
  return countParallelLoops(funcOp) > 0;
}

bool isFused(func::FuncOp funcOp) {
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
