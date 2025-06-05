# AutoPoly Unit Tests

find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${GTEST_INCLUDE_DIRS})

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

# Analysis Tests
add_executable(AnalysisTests
  Analysis/PolyhedralExtractionTest.cpp
  Analysis/DependenceAnalysisTest.cpp
)

target_link_libraries(AnalysisTests
  AutoPolyTestUtils
  ${GTEST_LIBRARIES}
  pthread
)

# Scheduling Tests
add_executable(SchedulingTests
  Scheduling/SchedulingStrategyTest.cpp
  Scheduling/SchedulingStrategyManagerTest.cpp
)

target_link_libraries(SchedulingTests
  AutoPolyTestUtils
  ${GTEST_LIBRARIES}
  pthread
)

# Transform Tests
add_executable(TransformTests
  Transform/TilingTransformTest.cpp
  Transform/ParallelizationTransformTest.cpp
  Transform/FusionTransformTest.cpp
  Transform/SkewingTransformTest.cpp
  Transform/SchedulingTransformerTest.cpp
)

target_link_libraries(TransformTests
  AutoPolyTestUtils
  ${GTEST_LIBRARIES}
  pthread
)

# CodeGen Tests
add_executable(CodeGenTests
  CodeGen/AffineCodeGenerationTest.cpp
  CodeGen/ParallelLoopOptimizerTest.cpp
  CodeGen/MemoryAccessOptimizerTest.cpp
)

target_link_libraries(CodeGenTests
  AutoPolyTestUtils
  ${GTEST_LIBRARIES}
  pthread
)

# Target Tests
add_executable(TargetTests
  Target/TargetDetectorTest.cpp
  Target/TargetCharacteristicsTest.cpp
)

target_link_libraries(TargetTests
  AutoPolyTestUtils
  ${GTEST_LIBRARIES}
  pthread
)

# Pass Tests
add_executable(PassTests
  Passes/AutoPolyPassesTest.cpp
  Passes/PolyhedralAnalysisPassTest.cpp
  Passes/DependenceAnalysisPassTest.cpp
  Passes/TargetDetectionPassTest.cpp
)

target_link_libraries(PassTests
  AutoPolyTestUtils
  ${GTEST_LIBRARIES}
  pthread
)

# Add test targets
add_custom_target(check-autopoly-unit
  COMMAND ${CMAKE_CTEST_COMMAND} --verbose
  DEPENDS AnalysisTests SchedulingTests TransformTests CodeGenTests TargetTests PassTests
)

# Register tests with CTest
enable_testing()
add_test(NAME AnalysisTests COMMAND AnalysisTests)
add_test(NAME SchedulingTests COMMAND SchedulingTests)
add_test(NAME TransformTests COMMAND TransformTests)
add_test(NAME CodeGenTests COMMAND CodeGenTests)
add_test(NAME TargetTests COMMAND TargetTests)
add_test(NAME PassTests COMMAND PassTests)
