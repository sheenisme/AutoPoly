cmake_minimum_required(VERSION 3.20.0)
project(AutoPoly LANGUAGES CXX C)

# Build with install name directory
# TODO: fix after support LLVM/Clang/MLIR in-tree build
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR OFF)

# Set C/C++ standards
set(CMAKE_C_STANDARD 17 CACHE STRING "C standard to conform to")
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_C_STANDARD_REQUIRED ON CACHE BOOL "Require C standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard")

# Set LLVM/Clang/MLIR build (out-of-tree or in-tree)
# TODO: fix in-tree build support method
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(STATUS "LLVM/Clang/MLIR out-of-tree build in ${LLVM_BUILD_DIR}.")

  # Set LLVM/Clang/MLIR cmake dir
  set(LLVM_DIR ${LLVM_BUILD_DIR}/lib/cmake/llvm)
  set(Clang_DIR ${LLVM_BUILD_DIR}/lib/cmake/clang)
  set(MLIR_DIR ${LLVM_BUILD_DIR}/lib/cmake/mlir)
  find_package(MLIR REQUIRED CONFIG)
  find_package(LLVM REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG)
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

  # Append LLVM/Clang/MLIR cmake dir
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  message(STATUS "Appended to CMAKE_MODULE_PATH: ${LLVM_CMAKE_DIR}")
  message(STATUS "Appended to CMAKE_MODULE_PATH: ${CLANG_CMAKE_DIR}")
  message(STATUS "Appended to CMAKE_MODULE_PATH: ${MLIR_CMAKE_DIR}")

  # Set LLVM/MLIR Library and binary dir
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  # Include required modules
  include(TableGen)
  include(AddLLVM)
  include(AddClang)
  include(AddMLIR)
  include(HandleLLVMOptions)

  # Set Clang Prefix and reqired Libs by PET
  set(PET_CLANG_PREFIX ${LLVM_BUILD_DIR})
  message(STATUS "PET_CLANG_PREFIX: ${PET_CLANG_PREFIX}")
  if(LLVM_FOUND)
    set(PET_REQUIRED_LLVM_LIBS
        LLVMSupport
        LLVMCore
        LLVMBinaryFormat
        LLVMRemarks
        LLVMBitstreamReader
        LLVMDemangle
        LLVMTargetParser
        LLVMTextAPI
    )
    list(APPEND CMAKE_REQUIRED_INCLUDES ${LLVM_INCLUDE_DIRS})
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${PET_REQUIRED_LLVM_LIBS})
    message(STATUS "Appended LLVM include: ${LLVM_INCLUDE_DIRS}")
    message(STATUS "Appended LLVM libraries: ${PET_REQUIRED_LLVM_LIBS}")
  else()
    message(FATAL_ERROR "LLVM not Found!!!")
  endif()
  if(Clang_FOUND)
    set(PET_REQUIRED_CLANG_LIBS
        clangBasic
        clangLex
        clangFrontend
        clangSerialization
        clangParse 
        clangSema
        clangAnalysis
        clangAST
        clangEdit
        clangASTMatchers
        clangDriver
        clangSupport
    )
    list(APPEND CMAKE_REQUIRED_INCLUDES ${CLANG_INCLUDE_DIRS})
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${PET_REQUIRED_CLANG_LIBS})
    message(STATUS "Appended Clang include: ${CLANG_INCLUDE_DIRS}")
    message(STATUS "Appended Clang libraries: ${PET_REQUIRED_CLANG_LIBS}")
  else()
    message(FATAL_ERROR "Clang not Found!!!")
  endif()
else()
  message(STATUS "LLVM/Clang/MLIR in-tree build.")
  
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(CLANG_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../clang)

  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
  set(LLVM_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/include)
  set(CLANG_INCLUDE_DIR ${CLANG_MAIN_SRC_DIR}/include)

  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
  include_directories(SYSTEM ${LLVM_INCLUDE_DIR})
  include_directories(SYSTEM ${CLANG_INCLUDE_DIR})

  # TODO: fix after support LLVM/Clang/MLIR in-tree build
  message(FATAL_ERROR "No Support LLVM/Clang/MLIR in-tree build!!!")
endif()

# General options
option(USE_MPFR   "Disable MPFR  support" OFF)
option(USE_OPENCL "Disable OpenCL support" OFF)
option(USE_OPENMP "Enable OpenMP support" ON)
option(BUILD_PET_EXE "Enable Build PET executable" ON)

# Set AutoPoly project directories
set(AUTOPOLY_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(AUTOPOLY_BINARY_DIR ${PROJECT_BINARY_DIR})

# Set default build type to Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Default type of build" FORCE)
endif()
message(STATUS "CMake Build type: ${CMAKE_BUILD_TYPE}")

# Set module search path for some dependencies tools
list(APPEND CMAKE_MODULE_PATH ${AUTOPOLY_SOURCE_DIR}/cmake)

# Configure required dependencies
include(ConfigDEPS)

# Configure all third-party libraries in ppcg directory
set(PPCG_DIR ${AUTOPOLY_SOURCE_DIR}/third_party/ppcg)
set(ISL_DIR ${PPCG_DIR}/isl)
include(ConfigISL)
set(PET_DIR ${PPCG_DIR}/pet)
include(ConfigPET)
include(ConfigPPCG)

# Configure AutoPoly application
message(STATUS "Configuring AutoPoly application")
set(AUTOPOLY_SOURCES
    ${AUTOPOLY_SOURCE_DIR}/lib/main.cpp
)
include_directories(AutoPoly PRIVATE
    ${AUTOPOLY_SOURCE_DIR}/include
    ${PPCG_DIR}
    ${ISL_DIR}/include
    ${PET_DIR}/include
)

# Build the main application
add_executable(AutoPoly ${AUTOPOLY_SOURCES})
set_target_properties(AutoPoly PROPERTIES OUTPUT_NAME "AutoPoly")
target_link_libraries(AutoPoly ppcg)

# Install AutoPoly application
install(TARGETS AutoPoly
    RUNTIME DESTINATION bin
)
