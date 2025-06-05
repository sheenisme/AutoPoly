#include "gtest/gtest.h"
#include "AutoPoly/Passes/AutoPolyPasses.h"
#include "TestUtils.h"

using namespace autopoly::passes;
using namespace autopoly::test;

class PassesTest : public PassTestBase {};

TEST_F(PassesTest, PipelineRegistration) {
  EXPECT_NO_THROW({
    registerAutoPolyPasses();
  });
}

TEST_F(PassesTest, PassCreation) {
  auto pass1 = createAutoPolySchedulingPass();
  auto pass2 = createPolyhedralAnalysisPass();
  auto pass3 = createDependenceAnalysisPass();
  auto pass4 = createTargetDetectionPass();
  EXPECT_TRUE(pass1 != nullptr);
  EXPECT_TRUE(pass2 != nullptr);
  EXPECT_TRUE(pass3 != nullptr);
  EXPECT_TRUE(pass4 != nullptr);
} 

// Use gtest main entry
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
