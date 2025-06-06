# AutoPoly Analysis Library
add_mlir_library(AutoPolyAnalysis
  PolyhedralExtraction.cpp
  DependenceAnalysis.cpp
  PolyhedralUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AUTOPOLY_SOURCE_DIR}/include/AutoPoly/Analysis

  DEPENDS
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRSupport
  MLIRIR
  MLIRMemRefDialect
  AutoPolyTarget
  ${PPCG_LIBRARIES}
)

target_include_directories(AutoPolyAnalysis
  PUBLIC
  ${AUTOPOLY_SOURCE_DIR}/include
  ${AUTOPOLY_BINARY_DIR}/include
  ${PPCG_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
)

target_link_libraries(AutoPolyAnalysis
  PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRSupport
  MLIRIR
  MLIRMemRefDialect
  MLIRDialect
  AutoPolyTarget
  ${PPCG_LIBRARIES}
)

# Install headers
install(
  DIRECTORY ${AUTOPOLY_SOURCE_DIR}/include/AutoPoly
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)
