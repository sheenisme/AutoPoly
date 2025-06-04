# AutoPoly Target Detection and Characterization Library

add_mlir_library(AutoPolyTarget
  TargetInfo.cpp
  TargetDetector.cpp
  CPUTarget.cpp
  GPUTarget.cpp
  # OpenCLTarget.cpp
  # FPGATarget.cpp
  # CGRATarget.cpp
  # NPUTarget.cpp

  ADDITIONAL_HEADER_DIRS
  ${AUTOPOLY_SOURCE_DIR}/include/AutoPoly/Target

  DEPENDS
  MLIRSupport
  MLIRIR
)

target_link_libraries(AutoPolyTarget
  PUBLIC
  MLIRSupport
  MLIRIR
  LLVMSupport
)

target_include_directories(AutoPolyTarget
  PUBLIC
  ${AUTOPOLY_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
