# AutoPoly Transformation Library

add_mlir_library(AutoPolyTransform
  SchedulingTransform.cpp
  # TilingTransformation.cpp
  # ParallelizationTransformation.cpp
  # FusionTransformation.cpp
  # SkewingTransformation.cpp
  # SchedulingTransformer.cpp
  # PPCGIntegration.cpp
  # TransformUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AUTOPOLY_SOURCE_DIR}/include/AutoPoly/Transform

  DEPENDS
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRSupport
  MLIRIR
  AutoPolyTarget
  AutoPolyScheduling
  AutoPolyAnalysis
  ${PPCG_LIBRARIES}
)

target_link_libraries(AutoPolyTransform
  PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRSupport
  MLIRIR
  AutoPolyTarget
  AutoPolyScheduling
  AutoPolyAnalysis
  ${PPCG_LIBRARIES}
)

target_include_directories(AutoPolyTransform
  PUBLIC
  ${AUTOPOLY_SOURCE_DIR}/include
  ${PPCG_INCLUDE_DIRS}
)
