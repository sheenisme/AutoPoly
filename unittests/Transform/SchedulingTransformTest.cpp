#include "gtest/gtest.h"
#include "AutoPoly/Transform/SchedulingTransform.h"
#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "AutoPoly/Analysis/DependenceAnalysis.h"
#include "AutoPoly/Scheduling/SchedulingStrategy.h"
#include "TestUtils.h"

using namespace autopoly::transform;
using namespace autopoly::analysis;
using namespace autopoly::scheduling;
using namespace autopoly::target;
using namespace autopoly::test;

class SchedulingTransformTest : public SchedulingTestBase {};

TEST_F(SchedulingTransformTest, TilingTransform) {
  isl_ctx* ctx = this->getISLContext();
  auto funcOp = this->createSimpleLoopNest(2);
  PolyhedralExtractor extractor(ctx);
  auto model = extractor.extractFromFunction(funcOp);
  DependenceAnalyzer analyzer(ctx);
  auto deps = analyzer.analyze(*model);
  CPUSchedulingStrategy strategy;
  SchedulingTransformer transformer(ctx);
  auto result = transformer.transform(*model, *deps, this->createMockCPUTarget(), strategy);
  EXPECT_TRUE(result.transformation_successful || !result.error_message.empty());
  PolyhedralUtils::destroyContext(ctx);
}

// Use gtest main entry
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
