add_executable(TargetUnitTests
  TargetDetectorTest.cpp
)

target_link_libraries(TargetUnitTests
  AutoPolyTestUtils
  gtest
  gtest_main
  pthread
)

add_test(NAME TargetUnitTests COMMAND TargetUnitTests)

# Remove self-dependency that causes cycle
# set_tests_properties(TargetUnitTests PROPERTIES DEPENDS TargetUnitTests)
