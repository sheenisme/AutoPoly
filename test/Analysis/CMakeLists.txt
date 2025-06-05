# Analysis tests for AutoPoly

# Add test files to the build system
configure_file(polyhedral-extraction.mlir polyhedral-extraction.mlir COPYONLY)
configure_file(dependence-analysis.mlir dependence-analysis.mlir COPYONLY)
configure_file(complex-affine.mlir complex-affine.mlir COPYONLY)

add_custom_target(AnalysisTests)
file(GLOB ANALYSIS_TESTS "*.mlir")
foreach(testfile ${ANALYSIS_TESTS})
  add_test(NAME analysis-${testfile}
           COMMAND autopoly-mlir-opt ${testfile} -autopoly-polyhedral-analysis)
endforeach()
