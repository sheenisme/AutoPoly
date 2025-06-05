#include "gtest/gtest.h"
#include "AutoPoly/Scheduling/SchedulingStrategy.h"
#include "AutoPoly/Target/TargetInfo.h"
#include "TestUtils.h"

using namespace autopoly::scheduling;
using namespace autopoly::target;
using namespace autopoly::test;

class SchedulingStrategyTest : public SchedulingTestBase {};

TEST_F(SchedulingStrategyTest, CPUStrategy) {
  CPUSchedulingStrategy strategy;
  auto cpu = this->createMockCPUTarget();
  EXPECT_TRUE(strategy.isSuitableForTarget(cpu));
  auto params = strategy.getParameters(cpu);
  EXPECT_TRUE(params.enable_tiling);
} 

// Use gtest main entry
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
