# AutoPoly Unit Tests

# Include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/unittests
  ${GTEST_INCLUDE_DIRS}
  ${PPCG_INCLUDE_DIRS}
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
  ${PPCG_LIBRARIES}
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
