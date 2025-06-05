# Passes tests for AutoPoly

add_custom_target(PassesTests)
file(GLOB PASSES_TESTS "*.mlir")
foreach(testfile ${PASSES_TESTS})
  add_test(NAME passes-${testfile}
           COMMAND autopoly-mlir-opt ${testfile} -autopoly-scheduling)
endforeach()