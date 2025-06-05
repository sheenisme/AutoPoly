add_executable(CodeGenUnitTests
  AffineCodeGenerationTest.cpp
)

target_link_libraries(CodeGenUnitTests
  AutoPolyTestUtils
  gtest
  gtest_main
  pthread
)

add_test(NAME CodeGenUnitTests COMMAND CodeGenUnitTests)
