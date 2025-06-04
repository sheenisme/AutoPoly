#include "AutoPoly/Analysis/PolyhedralExtractor.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace autopoly;

class PolyhedralExtractorTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.getOrLoadDialect<affine::AffineDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    
    builder = std::make_unique<OpBuilder>(&context);
    extractor = std::make_unique<PolyhedralExtractor>(&context);
  }
  
  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;
  std::unique_ptr<PolyhedralExtractor> extractor;
};

TEST_F(PolyhedralExtractorTest, SimpleForLoop) {
  // Create a simple for loop: for i = 0 to 10 { A[i] = B[i] + C[i] }
  Location loc = builder->getUnknownLoc();
  
  // Create memref types
  auto memrefType = MemRefType::get({100}, builder->getF32Type());
  
  // Create function
  auto funcType = builder->getFunctionType({memrefType, memrefType, memrefType}, {});
  auto funcOp = builder->create<func::FuncOp>(loc, "test_func", funcType);
  
  Block *entryBlock = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);
  
  Value A = entryBlock->getArgument(0);
  Value B = entryBlock->getArgument(1);
  Value C = entryBlock->getArgument(2);
  
  // Create for loop: for i = 0 to 10
  auto forOp = builder->create<affine::AffineForOp>(loc, 0, 10);
  builder->setInsertionPointToStart(forOp.getBody());
  
  Value iv = forOp.getInductionVar();
  
  // B[i]
  auto loadB = builder->create<affine::AffineLoadOp>(loc, B, iv);
  // C[i]  
  auto loadC = builder->create<affine::AffineLoadOp>(loc, C, iv);
  // B[i] + C[i]
  auto add = builder->create<arith::AddFOp>(loc, loadB, loadC);
  // A[i] = B[i] + C[i]
  builder->create<affine::AffineStoreOp>(loc, add, A, iv);
  
  builder->create<func::ReturnOp>(loc);
  
  // Extract polyhedral information
  auto polyInfo = extractor->extractFromAffineLoops(funcOp);
  
  ASSERT_NE(polyInfo, nullptr);
  EXPECT_FALSE(isl_union_set_is_empty(polyInfo->domain));
  EXPECT_FALSE(isl_union_map_is_empty(polyInfo->reads));
  EXPECT_FALSE(isl_union_map_is_empty(polyInfo->writes));
}

TEST_F(PolyhedralExtractorTest, NestedLoops) {
  // Create nested loops: for i = 0 to N { for j = 0 to M { A[i][j] = B[i][j] } }
  Location loc = builder->getUnknownLoc();
  
  auto memrefType = MemRefType::get({100, 100}, builder->getF32Type());
  auto funcType = builder->getFunctionType({memrefType, memrefType}, {});
  auto funcOp = builder->create<func::FuncOp>(loc, "nested_loops", funcType);
  
  Block *entryBlock = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);
  
  Value A = entryBlock->getArgument(0);
  Value B = entryBlock->getArgument(1);
  
  // Outer loop: for i = 0 to 100
  auto outerLoop = builder->create<affine::AffineForOp>(loc, 0, 100);
  builder->setInsertionPointToStart(outerLoop.getBody());
  Value i = outerLoop.getInductionVar();
  
  // Inner loop: for j = 0 to 100
  auto innerLoop = builder->create<affine::AffineForOp>(loc, 0, 100);
  builder->setInsertionPointToStart(innerLoop.getBody());
  Value j = innerLoop.getInductionVar();
  
  // B[i][j]
  auto loadB = builder->create<affine::AffineLoadOp>(loc, B, ValueRange{i, j});
  // A[i][j] = B[i][j]
  builder->create<affine::AffineStoreOp>(loc, loadB, A, ValueRange{i, j});
  
  builder->setInsertionPointAfter(innerLoop);
  builder->create<func::ReturnOp>(loc);
  
  // Extract and verify
  auto polyInfo = extractor->extractFromAffineLoops(funcOp);
  
  ASSERT_NE(polyInfo, nullptr);
  EXPECT_FALSE(isl_union_set_is_empty(polyInfo->domain));
  
  // Should have 2-dimensional iteration space
  int numDims = 0;
  isl_union_set_foreach_set(polyInfo->domain, [](isl_set *set, void *user) -> isl_stat {
    int *dims = static_cast<int*>(user);
    *dims = std::max(*dims, isl_set_dim(set, isl_dim_set));
    return isl_stat_ok;
  }, &numDims);
  
  EXPECT_EQ(numDims, 2);
}

TEST_F(PolyhedralExtractorTest, AffineIfCondition) {
  // Create loop with affine if: for i = 0 to N { if (i > 5) A[i] = B[i] }
  Location loc = builder->getUnknownLoc();
  
  auto memrefType = MemRefType::get({100}, builder->getF32Type());
  auto funcType = builder->getFunctionType({memrefType, memrefType}, {});
  auto funcOp = builder->create<func::FuncOp>(loc, "affine_if_test", funcType);
  
  Block *entryBlock = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);
  
  Value A = entryBlock->getArgument(0);
  Value B = entryBlock->getArgument(1);
  
  auto forOp = builder->create<affine::AffineForOp>(loc, 0, 100);
  builder->setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  
  // Create affine condition: i >= 6
  auto condition = AffineMap::get(1, 0, 
    builder->getAffineDimExpr(0) - builder->getAffineConstantExpr(6));
  auto integerSet = IntegerSet::get(1, 0, condition.getResult(), {false});
  
  auto ifOp = builder->create<affine::AffineIfOp>(loc, integerSet, ValueRange{iv}, false);
  builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
  
  auto loadB = builder->create<affine::AffineLoadOp>(loc, B, iv);
  builder->create<affine::AffineStoreOp>(loc, loadB, A, iv);
  
  builder->setInsertionPointAfter(ifOp);
  builder->create<func::ReturnOp>(loc);
  
  // Extract and verify conditional domain
  auto polyInfo = extractor->extractFromAffineLoops(funcOp);
  
  ASSERT_NE(polyInfo, nullptr);
  EXPECT_FALSE(isl_union_set_is_empty(polyInfo->domain));
}

TEST_F(PolyhedralExtractorTest, AffineParallelLoop) {
  // Create parallel loop: affine.parallel (%i) = (0) to (100) { A[%i] = B[%i] }
  Location loc = builder->getUnknownLoc();
  
  auto memrefType = MemRefType::get({100}, builder->getF32Type());
  auto funcType = builder->getFunctionType({memrefType, memrefType}, {});
  auto funcOp = builder->create<func::FuncOp>(loc, "parallel_test", funcType);
  
  Block *entryBlock = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);
  
  Value A = entryBlock->getArgument(0);
  Value B = entryBlock->getArgument(1);
  
  // Create parallel loop bounds
  auto lbMap = AffineMap::get(0, 0, builder->getAffineConstantExpr(0));
  auto ubMap = AffineMap::get(0, 0, builder->getAffineConstantExpr(100));
  
  auto parallelOp = builder->create<affine::AffineParallelOp>(
    loc, TypeRange{}, ValueRange{}, 
    ArrayRef<arith::AtomicRMWKind>{},
    ArrayRef<AffineMap>{lbMap}, ArrayRef<AffineMap>{ubMap},
    ArrayRef<int64_t>{1});
  
  Block *body = parallelOp.getBody();
  builder->setInsertionPointToStart(body);
  Value iv = body->getArgument(0);
  
  auto loadB = builder->create<affine::AffineLoadOp>(loc, B, iv);
  builder->create<affine::AffineStoreOp>(loc, loadB, A, iv);
  builder->create<affine::AffineYieldOp>(loc);
  
  builder->setInsertionPointAfter(parallelOp);
  builder->create<func::ReturnOp>(loc);
  
  // Extract and verify parallel loop
  auto polyInfo = extractor->extractFromAffineLoops(funcOp);
  
  ASSERT_NE(polyInfo, nullptr);
  EXPECT_FALSE(isl_union_set_is_empty(polyInfo->domain));
}

TEST_F(PolyhedralExtractorTest, ComplexAffineExpressions) {
  // Create loop with complex affine expressions: A[2*i + 3] = B[i + 5]
  Location loc = builder->getUnknownLoc();
  
  auto memrefType = MemRefType::get({200}, builder->getF32Type());
  auto funcType = builder->getFunctionType({memrefType, memrefType}, {});
  auto funcOp = builder->create<func::FuncOp>(loc, "complex_affine", funcType);
  
  Block *entryBlock = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);
  
  Value A = entryBlock->getArgument(0);
  Value B = entryBlock->getArgument(1);
  
  auto forOp = builder->create<affine::AffineForOp>(loc, 0, 50);
  builder->setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  
  // Create affine map for B[i + 5]
  auto readMap = AffineMap::get(1, 0, 
    builder->getAffineDimExpr(0) + builder->getAffineConstantExpr(5));
  auto loadB = builder->create<affine::AffineLoadOp>(loc, B, readMap, ValueRange{iv});
  
  // Create affine map for A[2*i + 3]
  auto writeMap = AffineMap::get(1, 0,
    builder->getAffineDimExpr(0) * 2 + builder->getAffineConstantExpr(3));
  builder->create<affine::AffineStoreOp>(loc, loadB, A, writeMap, ValueRange{iv});
  
  builder->create<func::ReturnOp>(loc);
  
  // Extract and verify complex expressions
  auto polyInfo = extractor->extractFromAffineLoops(funcOp);
  
  ASSERT_NE(polyInfo, nullptr);
  EXPECT_FALSE(isl_union_set_is_empty(polyInfo->domain));
  EXPECT_FALSE(isl_union_map_is_empty(polyInfo->reads));
  EXPECT_FALSE(isl_union_map_is_empty(polyInfo->writes));
}

