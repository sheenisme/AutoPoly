# FindClang.cmake
# This module can find Clang installation

# This module requires LLVM to have been found by FindLLVM.cmake
if(NOT LLVM_FOUND)
  message(FATAL_ERROR "Clang requires LLVM. Please find LLVM first.")
endif()

# If CLANG_ROOT is not set, try to use llvm-config to detect it
if(NOT DEFINED CLANG_ROOT)
  # Try to use llvm-config to find clang
  if(DEFINED LLVM_CONFIG)
    execute_process(
      COMMAND ${LLVM_CONFIG} --prefix
      OUTPUT_VARIABLE CLANG_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  else()
    # If LLVM_CONFIG is not defined, use LLVM_INSTALL_PREFIX
    set(CLANG_ROOT ${LLVM_INSTALL_PREFIX})
  endif()
endif()

# Find clang include directory
find_path(CLANG_INCLUDE_DIRS
  NAMES clang/Basic/Version.h
  PATHS
  ${CLANG_ROOT}/include
  /usr/lib/llvm-14/include
  /usr/lib/llvm-10/include
  /usr/include/llvm-14
  /usr/include/llvm-10
  NO_DEFAULT_PATH
)

# Find clang libraries directory
find_path(CLANG_LIBRARY_DIRS
  NAMES libclangBasic.a
  PATHS
  ${CLANG_ROOT}/lib
  /usr/lib/llvm-14/lib
  /usr/lib/llvm-10/lib
  NO_DEFAULT_PATH
)

# Find all clang libraries
set(CLANG_LIBRARIES "")
foreach(lib
    clangAST
    clangASTMatchers
    clangAnalysis
    clangBasic
    clangDriver
    clangEdit
    clangFrontend
    clangFrontendTool
    clangLex
    clangParse
    clangSema
    clangEdit
    clangRewrite
    clangRewriteFrontend
    clangStaticAnalyzerFrontend
    clangStaticAnalyzerCheckers
    clangStaticAnalyzerCore
    clangSerialization
    clangToolingCore
    clangTooling)
  find_library(CLANG_${lib}_LIBRARY
    NAMES ${lib}
    PATHS ${CLANG_LIBRARY_DIRS}
    NO_DEFAULT_PATH)
  if(CLANG_${lib}_LIBRARY)
    list(APPEND CLANG_LIBRARIES ${CLANG_${lib}_LIBRARY})
  endif()
endforeach()

# Handle the QUIETLY and REQUIRED arguments and set CLANG_FOUND to TRUE if all variables are not empty
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLANG DEFAULT_MSG CLANG_LIBRARIES CLANG_INCLUDE_DIRS)

mark_as_advanced(CLANG_INCLUDE_DIRS CLANG_LIBRARIES CLANG_LIBRARY_DIRS)