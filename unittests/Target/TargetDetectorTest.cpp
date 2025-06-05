#include "gtest/gtest.h"
#include "AutoPoly/Target/TargetInfo.h"
#include "TestUtils.h"

using namespace autopoly::target;
using namespace autopoly::test;

class TargetDetectorTest : public SchedulingTestBase {};

TEST_F(TargetDetectorTest, MockDetector) {
  TargetCharacteristics cpu = this->createMockCPUTarget();
  cpu.name = "MockCPU";
  std::vector<TargetCharacteristics> targets = {cpu};
  auto detector = TargetDetectorFactory::createMockDetector(targets);
  auto detected = detector->detectTargets();
  EXPECT_EQ(detected.size(), 1);
  EXPECT_EQ(detected[0].type, TargetType::CPU);
  EXPECT_TRUE(detector->isTargetAvailable(TargetType::CPU));
  EXPECT_EQ(detector->getTargetByName("MockCPU").name, "MockCPU");
}

TEST_F(TargetDetectorTest, SystemDetector) {
  auto detector = TargetDetectorFactory::createDetector();
  auto detected = detector->detectTargets();
  EXPECT_GE(detected.size(), 1);
  auto def = detector->getDefaultTarget();
  EXPECT_TRUE(def.name.size() > 0);
}

// Use gtest main entry
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
