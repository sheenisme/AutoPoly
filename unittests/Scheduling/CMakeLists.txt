add_executable(SchedulingUnitTests
  SchedulingStrategyTest.cpp
)

target_link_libraries(SchedulingUnitTests
  AutoPolyTestUtils
  gtest
  gtest_main
  pthread
)

add_test(NAME SchedulingUnitTests COMMAND SchedulingUnitTests)
