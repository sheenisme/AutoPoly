//===- TestUtils.h - AutoPoly Unit Test Utilities -------------------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file defines utility classes and functions for AutoPoly unit tests,
// providing common test infrastructure and helper methods.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOPOLY_UNITTESTS_TESTUTILS_H
#define AUTOPOLY_UNITTESTS_TESTUTILS_H

#include "AutoPoly/Analysis/DependenceAnalysis.h"
#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "AutoPoly/Target/TargetInfo.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include <gtest/gtest.h>
#include <memory>
#include <string>

// Forward declarations for ISL
struct isl_ctx;

namespace autopoly {
namespace test {

/// Base class for AutoPoly unit tests
class AutoPolyTestBase : public ::testing::Test {
protected:
  void SetUp() override;
  void TearDown() override;

  /// Get MLIR context
  mlir::MLIRContext* getMLIRContext() { return mlir_context_.get(); }
  
  /// Get ISL context
  isl_ctx* getISLContext() { return isl_context_; }
  
  /// Parse MLIR code from string
  mlir::OwningOpRef<mlir::ModuleOp> parseMLIR(const std::string& mlir_code);
  
  /// Create simple affine loop nest for testing
  mlir::func::FuncOp createSimpleLoopNest(int num_loops = 2);
  
  /// Create matrix multiplication function for testing
  mlir::func::FuncOp createMatMulFunction();
  
  /// Create complex affine IR with various constructs
  mlir::func::FuncOp createComplexAffineFunction();

private:
  std::unique_ptr<mlir::MLIRContext> mlir_context_;
  isl_ctx* isl_context_;
};

/// Test fixtures for specific components

/// Test fixture for analysis components
class AnalysisTestBase : public AutoPolyTestBase {
protected:
  void SetUp() override;
  
  /// Create polyhedral extractor
  std::unique_ptr<analysis::PolyhedralExtractor> createExtractor();
  
  /// Create dependence analyzer  
  std::unique_ptr<analysis::DependenceAnalyzer> createDependenceAnalyzer();
};

/// Test fixture for scheduling components
class SchedulingTestBase : public AutoPolyTestBase {
protected:
  void SetUp() override;
  
  /// Create mock target characteristics
  autopoly::target::TargetCharacteristics createMockCPUTarget();
  autopoly::target::TargetCharacteristics createMockGPUTarget();
  autopoly::target::TargetCharacteristics createMockFPGATarget();
  autopoly::target::TargetCharacteristics createMockNPUTarget();
};

/// Test fixture for transformation components
class TransformTestBase : public AutoPolyTestBase {
protected:
  void SetUp() override;
  
  /// Create sample polyhedral model for testing
  std::unique_ptr<analysis::PolyhedralModel> createSampleModel();
  
  /// Create sample dependence information
  std::unique_ptr<analysis::DependenceInfo> createSampleDependences();
};

/// Test fixture for code generation components
class CodeGenTestBase : public AutoPolyTestBase {
protected:
  void SetUp() override;
  
  /// Create sample ISL schedule for testing
  isl_schedule* createSampleSchedule();
  
  /// Verify generated MLIR operations
  bool verifyGeneratedOps(mlir::Operation* op);
};

/// Test fixture for pass components
class PassTestBase : public AutoPolyTestBase {
protected:
  void SetUp() override;
  
  /// Check if pass modified the function
  bool wasModified(mlir::func::FuncOp before, mlir::func::FuncOp after);
  /// Run a pass on a function
  mlir::LogicalResult runPassOnFunction(mlir::Pass* pass, mlir::func::FuncOp funcOp);
};

/// Utility functions for testing

/// Compare two MLIR operations for structural similarity
bool compareMLIROps(mlir::Operation* op1, mlir::Operation* op2);

/// Check if operation has specific attribute
bool hasAttribute(mlir::Operation* op, const std::string& attr_name);

/// Count loops in a function
int countLoops(mlir::func::FuncOp funcOp);

/// Count parallel loops in a function
int countParallelLoops(mlir::func::FuncOp funcOp);

/// Extract tile sizes from tiled loops
std::vector<int> extractTileSizes(mlir::func::FuncOp funcOp);

/// Check if function has been parallelized
bool isParallelized(mlir::func::FuncOp funcOp);

/// Check if function has been fused
bool isFused(mlir::func::FuncOp funcOp);

/// Mock classes for testing

/// Mock target detector for testing
class MockTargetDetector : public autopoly::target::TargetDetector {
public:
  MockTargetDetector();
  
  std::vector<autopoly::target::TargetCharacteristics> detectTargets() override;
  bool isTargetAvailable(autopoly::target::TargetType type) override;
  autopoly::target::TargetCharacteristics getDefaultTarget() override;
  
  /// Add mock target for testing
  void addMockTarget(const autopoly::target::TargetCharacteristics& target);
  
  /// Clear all mock targets
  void clearMockTargets();

private:
  std::vector<autopoly::target::TargetCharacteristics> mock_targets_;
};

/// Test data providers

/// Provides various MLIR test cases
class MLIRTestCases {
public:
  /// Simple loop nest
  static std::string getSimpleLoopNest();
  
  /// Matrix multiplication
  static std::string getMatrixMultiplication();
  
  /// Loop with conditional
  static std::string getLoopWithConditional();
  
  /// Nested loops with different bounds
  static std::string getNestedLoopsWithBounds();
  
  /// Loop with memory accesses
  static std::string getLoopWithMemoryAccesses();
  
  /// Complex affine expressions
  static std::string getComplexAffineExpressions();
  
  /// Multiple statement groups
  static std::string getMultipleStatementGroups();
  
  /// Function with return value
  static std::string getFunctionWithReturn();
  
  /// Function with parameters
  static std::string getFunctionWithParameters();
  
  /// Function with affine.yield
  static std::string getFunctionWithYield();
  
  /// Function with affine.if
  static std::string getFunctionWithIf();
};

/// Performance test utilities
class PerformanceTestUtils {
public:
  /// Measure execution time of a function
  static double measureExecutionTime(std::function<void()> func);
  
  /// Compare performance of two implementations
  static double comparePerformance(std::function<void()> baseline,
                                 std::function<void()> optimized);
  
  /// Generate performance report
  static std::string generatePerformanceReport(
      const std::map<std::string, double>& results);
};

} // namespace test
} // namespace autopoly

#endif // AUTOPOLY_UNITTESTS_TESTUTILS_H
