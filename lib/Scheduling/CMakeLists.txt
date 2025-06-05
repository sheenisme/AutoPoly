# AutoPoly Scheduling Strategy Library

add_mlir_library(AutoPolyScheduling
  SchedulingStrategy.cpp

  ADDITIONAL_HEADER_DIRS
  ${AUTOPOLY_SOURCE_DIR}/include/AutoPoly/Scheduling

  DEPENDS
  MLIRSupport
  MLIRIR
  AutoPolyTarget
  isl
  ppcg
)

target_link_libraries(AutoPolyScheduling
  PUBLIC
  MLIRSupport
  MLIRIR
  AutoPolyTarget
  isl
  ppcg
)

target_include_directories(AutoPolyScheduling
  PUBLIC
  ${AUTOPOLY_SOURCE_DIR}/include
  ${ISL_INCLUDE_DIRS}
  ${PPCG_INCLUDE_DIRS}
)
