# AutoPoly MLIR Passes Library

add_mlir_library(AutoPolyPasses
  AutoPolyPasses.cpp
  # AutoPolySchedulingPass.cpp
  # PolyhedralAnalysisPass.cpp
  # DependenceAnalysisPass.cpp
  # TargetDetectionPass.cpp
  # PassPipelineBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${AUTOPOLY_SOURCE_DIR}/include/AutoPoly/Passes

  DEPENDS
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRTransforms
  MLIRPass
  MLIRSupport
  MLIRIR
  AutoPolyTarget
  AutoPolyScheduling
  AutoPolyAnalysis
  AutoPolyTransform
  AutoPolyCodeGen
)

target_link_libraries(AutoPolyPasses
  PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRTransforms
  MLIRPass
  MLIRSupport
  MLIRIR
  AutoPolyTarget
  AutoPolyScheduling
  AutoPolyAnalysis
  AutoPolyTransform
  AutoPolyCodeGen
)

target_include_directories(AutoPolyPasses
  PUBLIC
  ${AUTOPOLY_SOURCE_DIR}/include
)
