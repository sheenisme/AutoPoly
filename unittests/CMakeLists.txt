# AutoPoly Unit Tests
# Add Google Test Framework
if(NOT TARGET GTest::gtest)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/googletest
    )
    FetchContent_MakeAvailable(googletest)
endif()
find_package(GTest REQUIRED)

# Include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/unittests
  ${GTEST_INCLUDE_DIRS}
  ${ISL_INCLUDE_DIRS}
  ${AUTOPOLY_BINARY_DIR}/include
  ${AUTOPOLY_BINARY_DIR}/include/isl
  ${AUTOPOLY_BINARY_DIR}/include/pet
  ${AUTOPOLY_BINARY_DIR}/include/ppcg
)

# Link directories
link_directories(${CMAKE_BINARY_DIR}/lib)

# Test utilities library
add_library(AutoPolyTestUtils
  TestUtils.cpp
)

target_link_libraries(AutoPolyTestUtils
  AutoPolyAnalysis
  AutoPolyScheduling
  AutoPolyTransform
  AutoPolyCodeGen
  AutoPolyTarget
  AutoPolyPasses
  ${GTEST_LIBRARIES}
  ${ISL_LIBRARIES}
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
)

# Add test subdirectories
add_subdirectory(Analysis)
add_subdirectory(CodeGen)
add_subdirectory(Scheduling)
add_subdirectory(Transform)
add_subdirectory(Target)
add_subdirectory(Passes)

# Add test targets
add_custom_target(check-autopoly-unit
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS
    PolyhedralExtractionTest
    DependenceAnalysisTest
    CodeGenUnitTests
    SchedulingUnitTests
    TransformUnitTests
    TargetUnitTests
    PassesUnitTests
)
