# AutoPoly Code Generation Library

add_mlir_library(AutoPolyCodeGen
  AffineCodeGeneration.cpp

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
  ${PPCG_LIBRARIES}
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
  ${PPCG_LIBRARIES}
)

target_include_directories(AutoPolyCodeGen
  PUBLIC
  ${AUTOPOLY_SOURCE_DIR}/include
  ${PPCG_INCLUDE_DIRS}
)
