# AutoPoly Tools

# Get all dialect and conversion libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# AutoPoly C optimizer (using PPCG directly)
add_executable(autopoly-c-opt autopoly-c-opt.cpp ${AUTOPOLY_SOURCE_DIR}/lib/ppcg_wrapper/ppcg_wrapper.c)
target_include_directories(autopoly-c-opt PRIVATE
  ${AUTOPOLY_SOURCE_DIR}/include
)
target_link_libraries(autopoly-c-opt PRIVATE ppcg pet isl)

# AutoPoly MLIR optimizer
set(AutoPolyMLIROPTDEPLIBS
  AutoPolyTarget
  AutoPolyScheduling
  AutoPolyAnalysis
  AutoPolyTransform
  AutoPolyPasses
  AutoPolyCodeGen
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRTransforms
  MLIRPass
  MLIRSupport
  MLIRIR
  ppcg
  isl
)
add_executable(autopoly-mlir-opt autopoly-mlir-opt.cpp)
target_link_libraries(autopoly-mlir-opt PRIVATE
  MLIROptLib
  ${AutoPolyMLIROPTDEPLIBS}
  ${dialect_libs}
  ${conversion_libs}
)

target_include_directories(autopoly-mlir-opt PRIVATE
  ${AUTOPOLY_SOURCE_DIR}/include
)
