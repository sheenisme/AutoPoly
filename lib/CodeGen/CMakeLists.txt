# AutoPoly Code Generation Library

add_mlir_library(AutoPolyCodeGen
  AffineCodeGeneration.cpp
  # ParallelLoopOptimizer.cpp
  # MemoryAccessOptimizer.cpp
  # CodeVerifier.cpp
  # CodeGenUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AUTOPOLY_SOURCE_DIR}/include/AutoPoly/CodeGen

  DEPENDS
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRSupport
  MLIRIR
  AutoPolyTarget
  AutoPolyAnalysis
  AutoPolyTransform
  isl
)

target_link_libraries(AutoPolyCodeGen
  PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRSupport
  MLIRIR
  AutoPolyTarget
  AutoPolyAnalysis
  AutoPolyTransform
  isl
)

target_include_directories(AutoPolyCodeGen
  PUBLIC
  ${AUTOPOLY_SOURCE_DIR}/include
  ${ISL_INCLUDE_DIRS}
)
