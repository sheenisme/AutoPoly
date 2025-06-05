add_executable(PassesUnitTests
  PassesTest.cpp
)

target_link_libraries(PassesUnitTests
  AutoPolyTestUtils
  gtest
  gtest_main
  pthread
)

add_test(NAME PassesUnitTests COMMAND PassesUnitTests)

# Remove self-dependency that causes cycle
# set_tests_properties(PassesUnitTests PROPERTIES DEPENDS PassesUnitTests)
