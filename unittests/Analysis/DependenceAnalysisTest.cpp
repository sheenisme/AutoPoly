#include "gtest/gtest.h"
#include "AutoPoly/Analysis/DependenceAnalysis.h"
#include "TestUtils.h"

using namespace autopoly::analysis;
using namespace autopoly::test;

class DependenceAnalysisTest : public AnalysisTestBase {};

TEST_F(DependenceAnalysisTest, BasicDependence) {
  isl_ctx* ctx = this->getISLContext();
  auto funcOp = this->createSimpleLoopNest(2);
  PolyhedralExtractor extractor(ctx);
  auto model = extractor.extractFromFunction(funcOp);
  DependenceAnalyzer analyzer(ctx);
  auto deps = analyzer.analyze(*model);
  ASSERT_TRUE(deps != nullptr);
  EXPECT_GE(deps->getDependences().size(), 0);
  PolyhedralUtils::destroyContext(ctx);
}

// Use gtest main entry
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
