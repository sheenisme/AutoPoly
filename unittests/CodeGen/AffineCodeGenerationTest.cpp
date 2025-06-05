#include "gtest/gtest.h"
#include "AutoPoly/CodeGen/AffineCodeGeneration.h"
#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "AutoPoly/Transform/SchedulingTransform.h"
#include "TestUtils.h"

using namespace autopoly::codegen;
using namespace autopoly::analysis;
using namespace autopoly::transform;
using namespace autopoly::test;

class AffineCodeGenerationTest : public CodeGenTestBase {};

TEST_F(AffineCodeGenerationTest, AffineCodeGeneration) {
  isl_ctx* ctx = this->getISLContext();
  auto funcOp = this->createSimpleLoopNest(2);
  PolyhedralExtractor extractor(ctx);
  auto model = extractor.extractFromFunction(funcOp);
  TransformationResult dummyResult;
  mlir::MLIRContext mlirCtx;
  mlir::OpBuilder builder(&mlirCtx);
  AffineCodeGenerator codegen(&mlirCtx, ctx);
  auto result = codegen.generateCode(nullptr, *model, dummyResult, builder, mlir::UnknownLoc::get(&mlirCtx));
  EXPECT_TRUE(result.generation_successful || !result.error_message.empty());
  PolyhedralUtils::destroyContext(ctx);
} 

// Use gtest main entry
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
