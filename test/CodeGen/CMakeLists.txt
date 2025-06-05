# CodeGen tests for AutoPoly

add_custom_target(CodeGenTests)
file(GLOB CODEGEN_TESTS "*.mlir")
foreach(testfile ${CODEGEN_TESTS})
  add_test(NAME codegen-${testfile}
           COMMAND autopoly-mlir-opt ${testfile} -autopoly-scheduling)
endforeach()