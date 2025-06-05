# Create separate executables for each test file
add_executable(PolyhedralExtractionTest
  PolyhedralExtractionTest.cpp
)

add_executable(DependenceAnalysisTest
  DependenceAnalysisTest.cpp
)

# Link libraries for both test executables
target_link_libraries(PolyhedralExtractionTest
  AutoPolyTestUtils
  gtest
  gtest_main
  pthread
)

target_link_libraries(DependenceAnalysisTest
  AutoPolyTestUtils
  gtest
  gtest_main
  pthread
)

# Add test targets
add_test(NAME PolyhedralExtractionTest COMMAND PolyhedralExtractionTest)
add_test(NAME DependenceAnalysisTest COMMAND DependenceAnalysisTest)

# Remove self-dependencies that cause cycles
# set_tests_properties(PolyhedralExtractionTest PROPERTIES DEPENDS PolyhedralExtractionTest)
# set_tests_properties(DependenceAnalysisTest PROPERTIES DEPENDS DependenceAnalysisTest)
