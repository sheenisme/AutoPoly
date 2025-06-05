add_executable(TransformUnitTests
  SchedulingTransformTest.cpp
)

target_link_libraries(TransformUnitTests
  AutoPolyTestUtils
  gtest
  gtest_main
  pthread
)

add_test(NAME TransformUnitTests COMMAND TransformUnitTests)
