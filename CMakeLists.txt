cmake_minimum_required(VERSION 3.10)
project(AutoStash LANGUAGES C CXX)

# Set C/C++ standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Force using system GCC at the beginning of CMake file
# TODO: Remove this once we have a proper way to detect the compiler like m4 in isl.
message(STATUS "Only Support the system GCC")
set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE STRING "C++ compiler" FORCE)
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# TODO: This is a hack to get the include paths(need to be fixed).
# Get compiler information, including default include paths
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -E -x c++ -v /dev/null
    ERROR_VARIABLE compiler_info
    OUTPUT_QUIET
)
# Parse output to get include paths - fix path filtering
string(REGEX MATCHALL "/usr/include[^\n]+|/usr/lib[^\n]+" include_paths "${compiler_info}")
# Add system standard library paths, only include valid directories
foreach(path ${include_paths})
    if(IS_DIRECTORY "${path}")
        include_directories(isl pet ppcg "${path}")
        message(STATUS "Adding include path: ${path}")
    endif()
endforeach()

# # Configure compiler by clang and clang++
# set(CMAKE_C_COMPILER clang CACHE STRING "C compiler" FORCE)
# set(CMAKE_CXX_COMPILER clang++ CACHE STRING "C++ compiler" FORCE)
# message(STATUS "C compiler location: ${CMAKE_C_COMPILER}")
# message(STATUS "C++ compiler location: ${CMAKE_CXX_COMPILER}")

# Show compiler search path
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-search-dirs
    OUTPUT_VARIABLE SEARCH_DIRS
)
message(STATUS "Compiler search path:\n${SEARCH_DIRS}")

# Get CXX stdlib path
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
    OUTPUT_VARIABLE C_PLUS_STD_LIB_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "C++ library location: ${C_PLUS_STD_LIB_PATH}")

# ============================================================================
# Compilation options (default: ON)
# ============================================================================
# General build options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(USE_MPFR   "Enable MPFR   support" OFF)
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_OPENCL "Enable OpenCL support" ON)

# Third-party library build options
# These options control whether to build libraries from source or use external installations
option(BUILD_ISL "Build ISL library from source" ON)
option(BUILD_PET "Build PET library from source" ON)
option(BUILD_PET_EXECUTABLE "Build PET executable" ON)

# External library paths (used when not building from source)
# If BUILD_ISL or BUILD_PET is OFF, these paths must be set to valid installation directories
set(EXTERNAL_ISL_DIR "" CACHE PATH "Path to external ISL installation directory")
set(EXTERNAL_PET_DIR "" CACHE PATH "Path to external PET installation directory")

# Notes:PPCG is always built from source (not configurable)
# This ensures compatibility with the project and prevents linking to incompatible versions


# Set module search path for some dependencies tools
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Detection of MPFR OpenMP and OpenCL
if(USE_MPFR)
    find_package(MPFR REQUIRED)
    if(MPFR_FOUND)
        message(STATUS "Found MPFR: ${MPFR_FOUND}")
        message(STATUS "Added MPFR include directories: ${MPFR_INCLUDE_DIRS}")
        message(STATUS "Added MPFR libraries: ${MPFR_LIBRARIES}")
        include_directories(${MPFR_INCLUDE_DIRS})
    else()
        message(STATUS "MPFR not found")
    endif()
endif()
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)    
        message(STATUS "Found OpenMP: ${OpenMP_FOUND}")
        message(STATUS "OpenMP include directories: ${OpenMP_INCLUDE_DIRS}")
        message(STATUS "OpenMP libraries: ${OpenMP_LIBRARIES}")
    else()
        message(STATUS "OpenMP not found")
    endif()
endif()
if(USE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        message(STATUS "Found OpenCL: ${OpenCL_FOUND}")
        message(STATUS "OpenCL include directories: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")
    else()
        message(STATUS "OpenCL not found")
    endif()
endif()

# Detect GMP Library
find_package(GMP REQUIRED)
if(GMP_FOUND)
    # TODO： Clearify the MP means, and update setup code to use GMP instead of IMATH.
    set(USE_GMP_FOR_MP ON)
    set(USE_IMATH_FOR_MP OFF)
    set(USE_SMALL_INT_OPT OFF)
    message(STATUS "Found GMP: ${GMP_FOUND}, USE_GMP_FOR_MP=${USE_GMP_FOR_MP}")
    message(STATUS "Added GMP include directories: ${GMP_INCLUDE_DIRS}")
    message(STATUS "Added GMP libraries: ${GMP_LIBRARIES}")
    include_directories(${GMP_INCLUDE_DIRS})
else()
    # TODO： Clearify the MP means, and update setup code to use GMP instead of IMATH.
    set(USE_GMP_FOR_MP OFF)
    set(USE_IMATH_FOR_MP ON)
    set(USE_SMALL_INT_OPT ON)
    message(STATUS "GMP not found, USE_GMP_FOR_MP=${USE_GMP_FOR_MP}")
endif()

# Add LLVM/Clang detection
find_package(LLVM REQUIRED CONFIG)
if(LLVM_FOUND)
    message(STATUS "Found LLVM: ${LLVM_FOUND}")
    message(STATUS "LLVM include directories: ${LLVM_INCLUDE_DIRS}")
    message(STATUS "LLVM libraries: ${LLVM_LIBRARIES}")
    include_directories(${LLVM_INCLUDE_DIRS})
else()
    message(STATUS "LLVM not found")
endif()
find_package(Clang REQUIRED CONFIG)
if(Clang_FOUND)
    message(STATUS "Found Clang: ${Clang_FOUND}")
    message(STATUS "Clang include directories: ${CLANG_INCLUDE_DIRS}")
    message(STATUS "Clang libraries: ${CLANG_LIBRARIES}")
    include_directories(${CLANG_INCLUDE_DIRS})
else()
    message(STATUS "Clang not found")
endif()

# Set default build type to Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
message(STATUS "Binary output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Library output directory: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "Archive output directory: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")

# Set default installation prefix
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# Set RPATH settings to ensure installed binaries can find shared libraries
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)


# ============================================================================
# Third-party library paths configuration
# ============================================================================

# Set PPCG directory (always required and built from source)
set(PPCG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ppcg)

# Determine compiler characteristics
# (These are referenced from Polly project configuration)
include(CheckCXXSourceCompiles)
include(CheckCSourceCompiles)
include(CheckIncludeFileCXX)
include(CheckSymbolExists)

# Like check_c_source_compiles, but sets the result to either
# 0 (error while compiling) or 1 (compiled successfully)
# Required for compatibility with autotool's AC_CHECK_DECLS
function (check_c_source_compiles_numeric _prog _var)
    check_c_source_compiles("${_prog}" "${_var}")
    if ("${${_var}}")
        set("${_var}" 1 PARENT_SCOPE)
    else()
        set("${_var}" 0 PARENT_SCOPE)
    endif ()
endfunction ()
function (check_cxx_source_compiles_numeric _prog _var)
    check_cxx_source_compiles("${_prog}" "${_var}")
    if ("${${_var}}")
        set("${_var}" 1 PARENT_SCOPE)
    else()
        set("${_var}" 0 PARENT_SCOPE)
    endif ()
endfunction ()

# Check for the existance of a type
function (check_c_type_exists _type _files _variable)
    set(_includes "")
    foreach (file_name ${_files})
        set(_includes "${_includes}#include<${file_name}>\n")
    endforeach()
    check_c_source_compiles("
    ${_includes}
    ${_type} typeVar;
    int main(void) {
    return 0;
    }
    " ${_variable})
endfunction ()

# ============================================================================
# ISL Library Configuration
# ============================================================================
# ISL (Integer Set Library) is a library for manipulating sets and relations of integer points
# bounded by linear constraints. It's a core dependency for PPCG and PET.
# Configure ISL library
if(BUILD_ISL)
    set(ISL_DIR ${PPCG_DIR}/isl)
    message(STATUS "Using ISL from source at: ${ISL_DIR}")

    ########################################################################
    #    Check compiler characteristics to set ISL compilation options     #
    #    (Referenced from Polly project configuration)                     #
    ########################################################################
    check_c_source_compiles("
    int func(void) __attribute__((__warn_unused_result__));
    int main(void) { return 0; }
    " HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
    set(GCC_WARN_UNUSED_RESULT)
    if (HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
        set(GCC_WARN_UNUSED_RESULT "__attribute__((__warn_unused_result__))")
    endif ()

    check_c_source_compiles("
    __attribute__ ((unused)) static void foo(void);
    int main(void) { return 0; }
    " HAVE___ATTRIBUTE__)

    check_c_source_compiles_numeric("
    #include <strings.h>
    int main(void) { (void)ffs(0); return 0; }
    " HAVE_DECL_FFS)

    check_c_source_compiles_numeric("
    int main(void) { (void)__builtin_ffs(0); return 0; }
    " HAVE_DECL___BUILTIN_FFS)

    check_c_source_compiles_numeric("
    #include <intrin.h>
    int main(void) { (void)_BitScanForward(NULL, 0); return 0; }
    " HAVE_DECL__BITSCANFORWARD)

    if (NOT HAVE_DECL_FFS AND
        NOT HAVE_DECL___BUILTIN_FFS AND
        NOT HAVE_DECL__BITSCANFORWARD)
        message(FATAL_ERROR "No ffs implementation found")
    endif ()

    check_c_source_compiles_numeric("
    #include <strings.h>
    int main(void) { (void)strcasecmp(\"\", \"\"); return 0; }
    " HAVE_DECL_STRCASECMP)

    check_c_source_compiles_numeric("
    #include <string.h>
    int main(void) { (void)_stricmp(\"\", \"\"); return 0; }
    " HAVE_DECL__STRICMP)

    if (NOT HAVE_DECL_STRCASECMP AND NOT HAVE_DECL__STRICMP)
        message(FATAL_ERROR "No strcasecmp implementation found")
    endif ()

    check_c_source_compiles_numeric("
    #include <strings.h>
    int main(void) { (void)strncasecmp(\"\", \"\", 0); return 0; }
    " HAVE_DECL_STRNCASECMP)

    check_c_source_compiles_numeric("
    #include <string.h>
    int main(void) { (void)_strnicmp(\"\", \"\", 0); return 0; }
    " HAVE_DECL__STRNICMP)

    if (NOT HAVE_DECL_STRNCASECMP AND NOT HAVE_DECL__STRNICMP)
        message(FATAL_ERROR "No strncasecmp implementation found")
    endif ()

    check_c_source_compiles_numeric("
    #include <stdio.h>
    int main(void) { snprintf((void*)0, 0, \" \"); return 0; }
    " HAVE_DECL_SNPRINTF)

    check_c_source_compiles_numeric("
    #include <stdio.h>
    int main(void) { _snprintf((void*)0, 0, \" \"); return 0; }
    " HAVE_DECL__SNPRINTF)

    if (NOT HAVE_DECL_SNPRINTF AND NOT HAVE_DECL__SNPRINTF)
        message(FATAL_ERROR "No snprintf implementation found")
    endif ()

    # Create stdint.h file (Referenced from Polly project configuration)
    check_c_type_exists(uint8_t "" HAVE_UINT8T)
    check_c_type_exists(uint8_t "stdint.h" HAVE_STDINT_H)
    check_c_type_exists(uint8_t "inttypes.h" HAVE_INTTYPES_H)
    check_c_type_exists(uint8_t "sys/types.h" HAVE_SYS_INTTYPES_H)
    if (HAVE_UINT8T)
        set(INCLUDE_STDINT_H "")
    elseif (HAVE_STDINT_H)
        set(INCLUDE_STDINT_H "#include <stdint.h>")
    elseif (HAVE_INTTYPES_H)
        set(INCLUDE_STDINT_H "#include <inttypes.h>")
    elseif (HAVE_SYS_INTTYPES_H)
        set(INCLUDE_STDINT_H "#include <sys/inttypes.h>")
    else()
        message(FATAL_ERROR "No stdint.h or compatible found")
    endif ()
    file(WRITE "${CMAKE_BINARY_DIR}/include/isl/stdint.h.tmp"
        "${INCLUDE_STDINT_H}\n")
    configure_file("${CMAKE_BINARY_DIR}/include/isl/stdint.h.tmp"
        "${CMAKE_BINARY_DIR}/include/isl/stdint.h" COPYONLY)
    
    # Create ISL configuration header file
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/isl_config.h.cmake" "${CMAKE_BINARY_DIR}/include/isl_config.h")
    include_directories(isl PUBLIC ${CMAKE_BINARY_DIR}/include)

    # Create gitversion.h file (required by ISL)
    execute_process(
        COMMAND git describe --always 
        WORKING_DIRECTORY ${ISL_DIR}
        OUTPUT_VARIABLE ISL_GIT_HEAD_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    file(WRITE ${ISL_DIR}/gitversion.h "#define GIT_HEAD_ID \"${ISL_GIT_HEAD_VERSION}\"")
    message(STATUS "ISL version: ${ISL_GIT_HEAD_VERSION}")

    # Include ISL headers directories
    include_directories(isl PUBLIC
        ${ISL_DIR}
        ${ISL_DIR}/imath
        ${ISL_DIR}/include
    )
else()
    if(EXTERNAL_ISL_DIR STREQUAL "")
        message(FATAL_ERROR "EXTERNAL_ISL_DIR must be set when BUILD_ISL is OFF")
    endif()
    set(ISL_DIR ${EXTERNAL_ISL_DIR})
    message(STATUS "Using external ISL from: ${ISL_DIR}")

    # Include ISL headers
    include_directories(isl PUBLIC
        ${ISL_DIR}
        ${ISL_DIR}/imath
        ${ISL_DIR}/include
    )
endif()

# Build ISL library
if(BUILD_ISL)
    # Build ISL from source
    message(STATUS "Configuring ISL library build from source")
    
    # ISL library source files - based on Polly project configuration
    set(ISL_SOURCES
        ${ISL_DIR}/isl_aff.c
        ${ISL_DIR}/isl_aff_map.c
        ${ISL_DIR}/isl_affine_hull.c
        ${ISL_DIR}/isl_arg.c
        ${ISL_DIR}/isl_ast.c
        ${ISL_DIR}/isl_ast_build.c
        ${ISL_DIR}/isl_ast_build_expr.c
        ${ISL_DIR}/isl_ast_codegen.c
        ${ISL_DIR}/isl_ast_graft.c
        ${ISL_DIR}/basis_reduction_tab.c
        ${ISL_DIR}/isl_bernstein.c
        ${ISL_DIR}/isl_blk.c
        ${ISL_DIR}/isl_bound.c
        ${ISL_DIR}/isl_box.c
        ${ISL_DIR}/isl_coalesce.c
        ${ISL_DIR}/isl_constraint.c
        ${ISL_DIR}/isl_convex_hull.c
        ${ISL_DIR}/isl_ctx.c
        ${ISL_DIR}/isl_deprecated.c
        ${ISL_DIR}/isl_dim_map.c
        ${ISL_DIR}/isl_equalities.c
        ${ISL_DIR}/isl_factorization.c
        ${ISL_DIR}/isl_farkas.c
        ${ISL_DIR}/isl_ffs.c
        ${ISL_DIR}/isl_flow.c
        ${ISL_DIR}/isl_fold.c
        ${ISL_DIR}/isl_hash.c
        ${ISL_DIR}/isl_id_to_ast_expr.c
        ${ISL_DIR}/isl_id_to_id.c
        ${ISL_DIR}/isl_id_to_pw_aff.c
        ${ISL_DIR}/isl_ilp.c
        ${ISL_DIR}/isl_input.c
        ${ISL_DIR}/isl_local.c
        ${ISL_DIR}/isl_local_space.c
        ${ISL_DIR}/isl_lp.c
        ${ISL_DIR}/isl_map.c
        ${ISL_DIR}/isl_map_list.c
        ${ISL_DIR}/isl_map_simplify.c
        ${ISL_DIR}/isl_map_subtract.c
        ${ISL_DIR}/isl_map_to_basic_set.c
        ${ISL_DIR}/isl_mat.c
        ${ISL_DIR}/isl_morph.c
        ${ISL_DIR}/isl_id.c
        ${ISL_DIR}/isl_obj.c
        ${ISL_DIR}/isl_options.c
        ${ISL_DIR}/isl_output.c
        ${ISL_DIR}/isl_point.c
        ${ISL_DIR}/isl_polynomial.c
        ${ISL_DIR}/isl_printer.c
        ${ISL_DIR}/print.c
        ${ISL_DIR}/isl_range.c
        ${ISL_DIR}/isl_reordering.c
        ${ISL_DIR}/isl_sample.c
        ${ISL_DIR}/isl_scan.c
        ${ISL_DIR}/isl_schedule.c
        ${ISL_DIR}/isl_schedule_band.c
        ${ISL_DIR}/isl_schedule_node.c
        ${ISL_DIR}/isl_schedule_read.c
        ${ISL_DIR}/isl_schedule_tree.c
        ${ISL_DIR}/isl_schedule_constraints.c
        ${ISL_DIR}/isl_scheduler.c
        ${ISL_DIR}/isl_scheduler_clustering.c
        ${ISL_DIR}/isl_scheduler_scc.c
        ${ISL_DIR}/isl_set_list.c
        ${ISL_DIR}/isl_sort.c
        ${ISL_DIR}/isl_space.c
        ${ISL_DIR}/isl_stream.c
        ${ISL_DIR}/isl_seq.c
        ${ISL_DIR}/isl_set_to_ast_graft_list.c
        ${ISL_DIR}/isl_stride.c
        ${ISL_DIR}/isl_tab.c
        ${ISL_DIR}/isl_tab_pip.c
        ${ISL_DIR}/isl_tarjan.c
        ${ISL_DIR}/isl_transitive_closure.c
        ${ISL_DIR}/isl_union_map.c
        ${ISL_DIR}/isl_val.c
        ${ISL_DIR}/isl_vec.c
        ${ISL_DIR}/isl_version.c
        ${ISL_DIR}/isl_vertices.c
        # GMP related files
        ${ISL_DIR}/isl_gmp.c
        ${ISL_DIR}/isl_val_gmp.c
        # Template implemented C files
        ${ISL_DIR}/set_to_map.c
        ${ISL_DIR}/set_from_map.c
        ${ISL_DIR}/uset_to_umap.c
        ${ISL_DIR}/uset_from_umap.c
    )
    
    # Build ISL library
    add_library(isl ${ISL_SOURCES})
    if(GMP_FOUND)
        target_link_libraries(isl ${GMP_LIBRARIES})
    endif()
    if(MPFR_FOUND)
        target_link_libraries(isl ${MPFR_LIBRARIES})
    endif()

    # Add debug options to ISL for better error diagnosis
    target_compile_options(isl PRIVATE -g)

    # Set installation rules for ISL
    install(TARGETS isl
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(DIRECTORY ${ISL_DIR}/include/ DESTINATION include/)
else()
    # Use external ISL library
    message(STATUS "Using external ISL library from: ${ISL_DIR}")
    
    # Find the ISL library in the specified directory
    find_library(ISL_LIBRARY isl PATHS ${ISL_DIR}/lib ${ISL_DIR}/lib64 NO_DEFAULT_PATH)
    
    if(NOT ISL_LIBRARY)
        message(FATAL_ERROR "Could not find ISL library in ${ISL_DIR}/lib or ${ISL_DIR}/lib64")
    endif()

    # Create imported target for ISL
    add_library(isl UNKNOWN IMPORTED)
    set_target_properties(isl PROPERTIES IMPORTED_LOCATION ${ISL_LIBRARY})
    message(STATUS "Found external ISL library: ${ISL_LIBRARY}")

    # Add include directories for external ISL
    include_directories(${ISL_DIR}/include)
endif()


# ============================================================================
# PET Library Configuration
# ============================================================================
# PET (Polyhedral Extraction Tool) is a library for extracting polyhedral models from C source code.
# It uses Clang for parsing C code and provides a polyhedral representation that can be used by PPCG.
if(BUILD_PET)
    set(PET_DIR ${PPCG_DIR}/pet)
    message(STATUS "Using PET from source at: ${PET_DIR}")

    # Create gitversion.h file (required by PET)
    execute_process(
        COMMAND git describe --always 
        WORKING_DIRECTORY ${PET_DIR}
        OUTPUT_VARIABLE PET_GIT_HEAD_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    file(WRITE ${PET_DIR}/gitversion.h "#define GIT_HEAD_ID \"${PET_GIT_HEAD_VERSION}\"")

    # Set PET version
    string(REGEX REPLACE "^pet-([0-9.]+).*" "\\1" PET_VERSION "${PET_GIT_HEAD_VERSION}")
    message(STATUS "PET version: ${PET_VERSION}")

    ########################################################################
    # Check compiler characteristics to set PET compilation options        #
    #       TODO： fix these codes.                                        #
    ########################################################################
    # Check for DiagnosticInfo/Diagnostic
    check_c_source_compiles("#include <clang/Basic/Diagnostic.h>\nint main() { return 0; }" HAVE_DIAGNOSTIC_H)
    if(HAVE_DIAGNOSTIC_H)
        check_c_source_compiles(
            "#include <clang/Basic/Diagnostic.h>\nint main() { clang::DiagnosticsEngine d(0, 0); return 0; }" 
            HAVE_DIAGNOSTICSENGINE)
        if(HAVE_DIAGNOSTICSENGINE)
            set(DiagnosticInfo "Diagnostic")
        endif()
    endif()
    
    # Check for HandleTopLevelDecl return type
    check_c_source_compiles(
        "#include <clang/AST/ASTConsumer.h>\nint main() { return 0; }" 
        HAVE_ASTCONSUMER_H)
    if(HAVE_ASTCONSUMER_H)
        check_c_source_compiles(
            "#include <clang/AST/ASTConsumer.h>\nclass C : public clang::ASTConsumer { bool HandleTopLevelDecl(clang::DeclGroupRef D) { return true; } };\nint main() { return 0; }" 
            HANDLE_TOPLEVEL_DECL_RETURNS_BOOL)
        if(HANDLE_TOPLEVEL_DECL_RETURNS_BOOL)
            set(HandleTopLevelDeclReturn "bool")
            set(HandleTopLevelDeclContinue "true")
        else()
            check_c_source_compiles(
                "#include <clang/AST/ASTConsumer.h>\nclass C : public clang::ASTConsumer { void HandleTopLevelDecl(clang::DeclGroupRef D) { } };\nint main() { return 0; }" 
                HANDLE_TOPLEVEL_DECL_RETURNS_VOID)
            if(HANDLE_TOPLEVEL_DECL_RETURNS_VOID)
                set(HandleTopLevelDeclReturn "void")
                # HandleTopLevelDeclContinue is not defined for void return type
            endif()
        endif()
    endif()
    
    # Check if TargetInfo::CreateTargetInfo takes shared_ptr
    check_c_source_compiles(
        "#include <clang/Basic/TargetInfo.h>\n#include <memory>\nint main() { clang::DiagnosticsEngine *Diags; std::shared_ptr<clang::TargetOptions> TO; clang::TargetInfo::CreateTargetInfo(*Diags, TO); return 0; }" 
        CREATETARGETINFO_TAKES_SHARED_PTR_CHECK)
    if(CREATETARGETINFO_TAKES_SHARED_PTR_CHECK)
        set(CREATETARGETINFO_TAKES_SHARED_PTR 1)
    endif()
    
    # Check if CompilerInstance::setInvocation takes shared_ptr
    check_c_source_compiles(
        "#include <clang/Frontend/CompilerInstance.h>\n#include <memory>\nint main() { clang::CompilerInstance *clang; clang::CompilerInvocation *invocation; clang->setInvocation(std::make_shared<clang::CompilerInvocation>(*invocation)); return 0; }" 
        SETINVOCATION_TAKES_SHARED_PTR_CHECK)
    if(SETINVOCATION_TAKES_SHARED_PTR_CHECK)
        set(SETINVOCATION_TAKES_SHARED_PTR 1)
    endif()
    
    # Check if HeaderSearchOptions::AddPath takes 4 arguments
    check_c_source_compiles(
        "#include <clang/Lex/HeaderSearchOptions.h>\nint main() { clang::HeaderSearchOptions HSO; HSO.AddPath(\"\", clang::frontend::Angled, false, false); return 0; }" 
        ADDPATH_TAKES_4_ARGUMENTS_CHECK)
    if(ADDPATH_TAKES_4_ARGUMENTS_CHECK)
        set(ADDPATH_TAKES_4_ARGUMENTS 1)
    endif()
    
    # Check for ext_implicit_function_decl_c99
    check_c_source_compiles(
        "#include <clang/Basic/DiagnosticCategories.h>\nint main() { clang::diag::kind k = clang::diag::ext_implicit_function_decl_c99; return 0; }" 
        HAVE_EXT_IMPLICIT_FUNCTION_DECL_C99)
    if(NOT HAVE_EXT_IMPLICIT_FUNCTION_DECL_C99)
        set(ext_implicit_function_decl_c99 "ext_implicit_function_decl")
    endif()
    
    # Check for translateLineCol method
    check_c_source_compiles(
        "#include <clang/Basic/SourceManager.h>\nint main() { clang::SourceManager SM(clang::DiagnosticsEngine()); SM.translateLineCol(clang::FileID(), 0, 0); return 0; }" 
        HAVE_TRANSLATELINECOL_CHECK)
    if(HAVE_TRANSLATELINECOL_CHECK)
        set(HAVE_TRANSLATELINECOL 1)
    endif()
    
    # Check for standard C functions
    check_c_source_compiles_numeric(
        "#include <strings.h>\nint main() { (void)ffs(0); return 0; }" 
        HAVE_DECL_FFS)
    
    check_c_source_compiles_numeric(
        "#include <strings.h>\nint main() { (void)strcasecmp(\"\", \"\"); return 0; }" 
        HAVE_DECL_STRCASECMP)
    
    check_c_source_compiles_numeric(
        "#include <strings.h>\nint main() { (void)strncasecmp(\"\", \"\", 0); return 0; }" 
        HAVE_DECL_STRNCASECMP)

    # Check for getrusage and gettimeofday
    check_c_source_compiles(
        "#include <sys/resource.h>\nint main() { struct rusage r; getrusage(0, &r); return 0; }" 
        HAVE_GETRUSAGE_CHECK)
    if(HAVE_GETRUSAGE_CHECK)
        set(HAVE_GETRUSAGE 1)
    endif()
    
    check_c_source_compiles(
        "#include <sys/time.h>\nint main() { struct timeval t; gettimeofday(&t, 0); return 0; }" 
        HAVE_GETTIMEOFDAY_CHECK)
    if(HAVE_GETTIMEOFDAY_CHECK)
        set(HAVE_GETTIMEOFDAY 1)
    endif()
    
    # Check for ssize_t type
    check_c_type_exists(ssize_t "sys/types.h" HAVE_SSIZE_T_CHECK)
    if(HAVE_SSIZE_T_CHECK)
        set(HAVE_SSIZE_T 1)
    endif()
    
    # Check for stdint.h
    check_c_source_compiles(
        "#include <stdint.h>\nint main() { uint8_t x = 0; return 0; }" 
        HAVE_STDINT_H_CHECK)
    if(HAVE_STDINT_H_CHECK)
        set(HAVE_STDINT_H 1)
    endif()
    
    # Check for nested array size modifier
    check_c_source_compiles(
        "#include <clang/AST/Type.h>\nint main() { clang::ArrayType::ArraySizeModifier m; return 0; }" 
        USE_NESTED_ARRAY_SIZE_MODIFIER_CHECK)
    if(NOT USE_NESTED_ARRAY_SIZE_MODIFIER_CHECK)
        set(USE_NESTED_ARRAY_SIZE_MODIFIER 1)
    endif()
    
    # Check for getBeginLoc and getEndLoc
    check_c_source_compiles(
        "#include <clang/AST/Decl.h>\nint main() { clang::FunctionDecl *fd; fd->getBeginLoc(); fd->getEndLoc(); return 0; }" 
        HAVE_BEGIN_END_LOC_CHECK)
    if(HAVE_BEGIN_END_LOC_CHECK)
        set(HAVE_BEGIN_END_LOC 1)
    endif()
    
    # Check for getLocWithOffset vs getFileLocWithOffset
    check_c_source_compiles(
        "#include <clang/Basic/SourceLocation.h>\nint main() { clang::SourceLocation L; L.getLocWithOffset(0); return 0; }" 
        HAVE_GETLOCWITHOFFSET)
    if(NOT HAVE_GETLOCWITHOFFSET)
        set(getFileLocWithOffset "getLocWithOffset")
    endif()

    # # Check if getTypeInfo returns TypeInfo object
    # check_c_source_compiles(
    #     "#include <clang/AST/ASTContext.h>\nint main() { clang::ASTContext C(clang::LangOptions(), nullptr); clang::TypeInfo TI = C.getTypeInfo(nullptr); return 0; }" 
    #     GETTYPEINFORETURNSTYPEINFO_CHECK)
    # if(GETTYPEINFORETURNSTYPEINFO_CHECK)
    #     set(GETTYPEINFORETURNSTYPEINFO 1)
    # endif()
    
    # Check for getTypeInfo return type
    check_cxx_source_compiles_numeric("
    #include <clang/AST/ASTContext.h>
    int main(void) { clang::ASTContext.getTypeInfo(0).Width; return 0; }"
        GETTYPEINFORETURNSTYPEINFO
    )

    # Check for various header files
    check_include_file_cxx("clang/Basic/DiagnosticOptions.h" HAVE_BASIC_DIAGNOSTICOPTIONS_H)
    check_include_file_cxx("clang/Lex/HeaderSearchOptions.h" HAVE_LEX_HEADERSEARCHOPTIONS_H)
    check_include_file_cxx("clang/Basic/LangStandard.h" HAVE_CLANG_BASIC_LANGSTANDARD_H)
    check_include_file_cxx("clang/Lex/PreprocessorOptions.h" HAVE_LEX_PREPROCESSOROPTIONS_H)
    check_include_file_cxx("llvm/Option/Arg.h" HAVE_LLVM_OPTION_ARG_H)
    check_include_file_cxx("llvm/TargetParser/Host.h" HAVE_TARGETPARSER_HOST_H)

    # Check if createPreprocessor takes TranslationUnitKind
    check_c_source_compiles(
        "#include <clang/Frontend/CompilerInstance.h>\nint main() { clang::CompilerInstance *clang; clang->createPreprocessor(clang::TU_Complete); return 0; }" 
        CREATEPREPROCESSOR_TAKES_TUKIND_CHECK)
    if(CREATEPREPROCESSOR_TAKES_TUKIND_CHECK)
        set(CREATEPREPROCESSOR_TAKES_TUKIND 1)
    endif()
    
    # Check for setMainFileID method
    check_c_source_compiles(
        "#include <clang/Basic/SourceManager.h>\nint main() { return 0; }" 
        HAVE_SETMAINFILEID_CHECK)
    if(HAVE_SETMAINFILEID_CHECK)
        set(HAVE_SETMAINFILEID 1)
    endif()
    
    # Check if setLangDefaults takes 5 arguments
    check_c_source_compiles(
        "#include <clang/Basic/TargetOptions.h>\n#include <clang/Lex/PreprocessorOptions.h>\n#include <clang/Frontend/CompilerInstance.h>\nint main() { return 0; }" 
        SETLANGDEFAULTS_TAKES_5_ARGUMENTS_CHECK)
    if(SETLANGDEFAULTS_TAKES_5_ARGUMENTS_CHECK)
        set(SETLANGDEFAULTS_TAKES_5_ARGUMENTS 1)
    endif()
    
    # Check for class with setLangDefaults method
    check_c_source_compiles(
        "#include <clang/Basic/LangOptions.h>\nint main() { clang::LangOptions::setLangDefaults(clang::LangOptions(), clang::IK_C, llvm::Triple(), clang::PreprocessorOptions(), clang::LangStandard::lang_unspecified); return 0; }" 
        SETLANGDEFAULTS_IS_LANGOPTIONS)
    if(SETLANGDEFAULTS_IS_LANGOPTIONS)
        set(SETLANGDEFAULTS "LangOptions")
    else()
        set(SETLANGDEFAULTS "CompilerInvocation")
    endif()
    
    # Check for IK_C definition
    check_c_source_compiles(
        "#include <clang/Frontend/FrontendOptions.h>\nint main() { clang::InputKind::Language L = clang::Language::C; return 0; }" 
        IK_C_IS_LANGUAGE_C)
    if(IK_C_IS_LANGUAGE_C)
        set(IK_C "Language::C")
    else()
        check_c_source_compiles(
            "#include <clang/Frontend/FrontendOptions.h>\nint main() { clang::InputKind::Language L = clang::InputKind::C; return 0; }" 
            IK_C_IS_INPUTKIND_C)
        if(IK_C_IS_INPUTKIND_C)
            set(IK_C "InputKind::C")
        else()
            set(IK_C "IK_C")
        endif()
    endif()
    # Create PET configuration header file using the template
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/pet_config.h.cmake" "${CMAKE_BINARY_DIR}/include/pet/auto_config.h")


    # Create PET configuration header file using the hard codes.
    #   (TODO: this need by replace by auto_config.h)
    # configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/pet_config.h" "${CMAKE_BINARY_DIR}/include/pet/config.h")
    file(WRITE ${CMAKE_BINARY_DIR}/include/pet/config.h "\n/* Configuration file defining macros needed by the PET library */\n#define DiagnosticInfo Diagnostic\n#define HandleTopLevelDeclReturn bool\n#define HandleTopLevelDeclContinue true\n#define CREATETARGETINFO_TAKES_SHARED_PTR 1\n#define SETINVOCATION_TAKES_SHARED_PTR\n#define ADDPATH_TAKES_4_ARGUMENTS\n#define ext_implicit_function_decl_c99 ext_implicit_function_decl\n#define HAVE_TRANSLATELINECOL 1\n#define HAVE_DECL_FFS 1\n#define HAVE_DECL_STRCASECMP 1\n#define HAVE_DECL_STRNCASECMP 1\n#define HAVE_GETRUSAGE 1\n#define HAVE_GETTIMEOFDAY 1\n#define HAVE_SSIZE_T 1\n#define PACKAGE \"pet\"\n#define PACKAGE_BUGREPORT \"\"\n#define PACKAGE_NAME \"pet\"\n#define PACKAGE_STRING \"pet 0.11\"\n#define PACKAGE_TARNAME \"pet\"\n#define PACKAGE_URL \"\"\n#define PACKAGE_VERSION \"0.11\"\n#define VERSION \"0.11\"\n#define HAVE_STDINT_H 1\n#define USE_NESTED_ARRAY_SIZE_MODIFIER 1\n#define USE_GET_BEGIN_END_LOC 1\n#define getFileLocWithOffset getLocWithOffset\n#define GETTYPEINFORETURNSTYPEINFO 1\n#define HAVE_BASIC_DIAGNOSTICOPTIONS_H 1\n#define HAVE_LEX_HEADERSEARCHOPTIONS_H 1\n#define HAVE_CLANG_BASIC_LANGSTANDARD_H 1\n#define HAVE_LEX_PREPROCESSOROPTIONS_H 1\n#define HAVE_LLVM_OPTION_ARG_H 1\n#define CREATEPREPROCESSOR_TAKES_TUKIND 1\n#define HAVE_SETMAINFILEID 1\n#define SETLANGDEFAULTS_TAKES_5_ARGUMENTS 1\n#define SETLANGDEFAULTS CompilerInvocation\n#define IK_C Language::C\n#define HAVE_BEGIN_END_LOC 1\n")

    include_directories(pet PRIVATE ${CMAKE_BINARY_DIR}/include/pet)

    # include PET headers directories
    include_directories(pet PRIVATE
        ${CMAKE_BINARY_DIR}/include/pet
        ${PET_DIR}/include
        ${ISL_DIR}/include
    )

    # Add Clang dependency libraries
    set(CLANG_REQUIRED_LIBS
        clangAST
        clangBasic
        clangLex
        clangSerialization
        clangFrontend
        clangDriver
        clangParse
        clangSema
        clangAnalysis
        clangEdit
        clangASTMatchers
    )
else()
    if(EXTERNAL_PET_DIR STREQUAL "")
        message(FATAL_ERROR "EXTERNAL_PET_DIR must be set when BUILD_PET is OFF")
    endif()
    set(PET_DIR ${EXTERNAL_PET_DIR})
    message(STATUS "Using external PET from: ${PET_DIR}")
endif()

if(BUILD_PET)
    # Build PET from source
    message(STATUS "Configuring PET library build from source")

    # Complete PET source file list (according to Makefile.am)
    set(PET_SOURCES
        ${PET_DIR}/aff.c
        ${PET_DIR}/array.c
        ${PET_DIR}/clang.cc
        ${PET_DIR}/context.c
        ${PET_DIR}/expr.c
        ${PET_DIR}/expr_arg.c
        ${PET_DIR}/expr_plus.cc
        ${PET_DIR}/filter.c
        ${PET_DIR}/id.cc
        ${PET_DIR}/isl_id_to_pet_expr.c
        ${PET_DIR}/inlined_calls.cc
        ${PET_DIR}/inliner.cc
        ${PET_DIR}/killed_locals.cc
        ${PET_DIR}/loc.c
        ${PET_DIR}/nest.c
        ${PET_DIR}/options.c
        ${PET_DIR}/patch.c
        ${PET_DIR}/pet_expr_to_isl_pw_aff.c
        ${PET_DIR}/print.c
        ${PET_DIR}/tree.c
        ${PET_DIR}/tree2scop.c
        ${PET_DIR}/scan.cc
        ${PET_DIR}/scop.c
        ${PET_DIR}/scop_plus.cc
        ${PET_DIR}/skip.c
        ${PET_DIR}/substituter.cc
        ${PET_DIR}/summary.c
        ${PET_DIR}/value_bounds.c
        ${PET_DIR}/version.cc
        ${PET_DIR}/pet.cc
    )

    # Build pet library
    add_library(pet ${PET_SOURCES})

    # Add include directories for PET
    if(LLVM_FOUND)
        target_include_directories(pet PRIVATE ${LLVM_INCLUDE_DIRS})
    endif()
    if(Clang_FOUND)
        target_include_directories(pet PRIVATE ${CLANG_INCLUDE_DIRS})
    endif()
    
    # Add required libraries for PET (TODO: fix errors when clang not found)
    target_link_libraries(pet isl ${CLANG_REQUIRED_LIBS})
    if(GMP_FOUND)
        target_link_libraries(pet ${GMP_LIBRARIES})
    endif()
    if(MPFR_FOUND)
        target_link_libraries(pet ${MPFR_LIBRARIES})
    endif()
    if(LLVM_FOUND)
        target_link_libraries(pet ${LLVM_LIBRARIES})
    endif()
    if(Clang_FOUND)
        target_link_libraries(pet ${CLANG_LIBRARIES})
    endif()

    # Set installation rules for PET
    install(TARGETS pet
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(DIRECTORY ${PET_DIR}/include/ DESTINATION include/pet)
    
    # Build PET executable
    if(BUILD_PET_EXECUTABLE)
        message(STATUS "Configuring PET executable build from source")
        # Check libyaml library (required by pet executable)
        find_package(YAML REQUIRED)
        if(YAML_FOUND)
            message(STATUS "LibYaml library found")
            message(STATUS "LibYaml include directories: ${YAML_INCLUDE_DIRS}")
            message(STATUS "LibYaml libraries: ${YAML_LIBRARIES}")
        else ()
            message(FATAL_ERROR "LibYaml library not found")
        endif()

        # Build PET executable from source
        add_executable(pet_exe
            ${PET_DIR}/main.c
            ${PET_DIR}/dummy.cc
            ${PET_DIR}/emit.c
        )
        set_target_properties(pet_exe PROPERTIES OUTPUT_NAME "pet")
        target_include_directories(pet_exe PRIVATE
            ${YAML_INCLUDE_DIRS}
        )

        # Link PET executable with required libraries
        target_link_libraries(pet_exe PRIVATE
            pet
            isl
            ${CLANG_REQUIRED_LIBS}
            ${YAML_LIBRARIES}
        )
        if(GMP_FOUND)
            target_link_libraries(pet_exe PRIVATE ${GMP_LIBRARIES})
        endif()
        if(MPFR_FOUND)
            target_link_libraries(pet_exe PRIVATE ${MPFR_LIBRARIES})
        endif()
        if(LLVM_FOUND)
            target_link_libraries(pet_exe PRIVATE ${LLVM_LIBRARIES})
        endif()
        if(Clang_FOUND)
            target_link_libraries(pet_exe PRIVATE ${CLANG_LIBRARIES})
        endif()

        # Set installation rules for PET_EXECUTABLE
        install(TARGETS pet_exe
            RUNTIME DESTINATION bin
        )
    endif()
else()
    # Use external PET library
    message(STATUS "Using external PET library from: ${PET_DIR}")
    
    # Find the PET library in the specified directory
    find_library(PET_LIBRARY pet PATHS ${PET_DIR}/lib ${PET_DIR}/lib64 NO_DEFAULT_PATH)
    
    if(NOT PET_LIBRARY)
        message(FATAL_ERROR "Could not find PET library in ${PET_DIR}/lib or ${PET_DIR}/lib64")
    endif()
    
    # Create imported target for PET
    add_library(pet UNKNOWN IMPORTED)
    set_target_properties(pet PROPERTIES IMPORTED_LOCATION ${PET_LIBRARY})
    message(STATUS "Found external PET library: ${PET_LIBRARY}")
    
    # Add include directories for external PET
    include_directories(${PET_DIR}/include)
endif()

# ============================================================================
# PPCG Library Configuration (Always built from source)
# ============================================================================
# PPCG (Polyhedral Parallel Code Generator) is the core library for this project.
# It transforms C code to parallel code using the polyhedral model.
# PPCG is ALWAYS built from source to ensure compatibility with the project.
# No external linking option is provided to prevent version incompatibilities.
message(STATUS "Configuring PPCG library build from source (required)")

# Create gitversion.h file (required by PPCG)
execute_process(
    COMMAND git describe --always 
    WORKING_DIRECTORY ${PPCG_DIR}
    OUTPUT_VARIABLE PPCG_GIT_HEAD_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(WRITE ${PPCG_DIR}/gitversion.h "#define GIT_HEAD_ID \"${PPCG_GIT_HEAD_VERSION}\"")
message(STATUS "PPCG version: ${PPCG_GIT_HEAD_VERSION}")  

include_directories(
    ${PPCG_DIR}
    ${ISL_DIR}/include
    ${PET_DIR}/include
)

set(PPCG_SOURCES
    ${PPCG_DIR}/cpu.c
    ${PPCG_DIR}/cuda.c
    ${PPCG_DIR}/opencl.c
    ${PPCG_DIR}/cuda_common.c
    ${PPCG_DIR}/gpu.c
    ${PPCG_DIR}/gpu_array_tile.c
    ${PPCG_DIR}/gpu_group.c
    ${PPCG_DIR}/gpu_hybrid.c
    ${PPCG_DIR}/gpu_print.c
    ${PPCG_DIR}/gpu_tree.c
    ${PPCG_DIR}/grouping.c
    ${PPCG_DIR}/hybrid.c
    ${PPCG_DIR}/schedule.c
    ${PPCG_DIR}/ppcg_options.c
    ${PPCG_DIR}/print.c
    ${PPCG_DIR}/util.c
    ${PPCG_DIR}/version.c
    # Remove ppcg.c from the library to avoid main function conflict
    # ${PPCG_DIR}/ppcg.c
    # Add ppcg_wrapper.c file containing ppcg_main implementation
    ${CMAKE_CURRENT_SOURCE_DIR}/ppcg_wrapper.c
)

# Build PPCG library
add_library(ppcg ${PPCG_SOURCES})
target_link_libraries(ppcg pet isl)
if(GMP_FOUND)
    target_link_libraries(ppcg ${GMP_LIBRARIES})
endif()
if(MPFR_FOUND)
    target_link_libraries(ppcg ${MPFR_LIBRARIES})
endif()
if(OpenMP_FOUND)
    target_link_libraries(ppcg ${OpenMP_CXX_LIBRARIES})
endif()
if(OpenCL_FOUND)
    target_link_libraries(ppcg ${OpenCL_LIBRARIES})
endif()

# Build PPCG executable
add_executable(ppcg_exe ${PPCG_DIR}/ppcg.c)
set_target_properties(ppcg_exe PROPERTIES OUTPUT_NAME "ppcg")
target_link_libraries(ppcg_exe PRIVATE ppcg pet isl)
if(GMP_FOUND)
    target_link_libraries(ppcg_exe PRIVATE ${GMP_LIBRARIES})
endif()
if(MPFR_FOUND)
    target_link_libraries(ppcg_exe PRIVATE ${MPFR_LIBRARIES})
endif()
if(OpenMP_FOUND)
    target_link_libraries(ppcg_exe PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()
if(OpenCL_FOUND)
    target_link_libraries(ppcg_exe PRIVATE ${OpenCL_LIBRARIES})
endif()

# Set installation rules for PPCG
install(TARGETS ppcg ppcg_exe
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install PPCG headers
file(GLOB PPCG_HEADERS "${PPCG_DIR}/*.h") 
install(FILES ${PPCG_HEADERS} 
    DESTINATION include/ppcg
)



# ============================================================================
# Note on third-party library installation:
# - PPCG: Always installed (required component, built from source)
# - ISL: Installed only if BUILD_ISL=ON (built from source)
# - PET: Installed only if BUILD_PET=ON (built from source)
# 
# Each library's installation rules are defined in their respective sections above.
# This separation makes it clear which components belong to the project and which
# are third-party dependencies.
# ============================================================================


# ============================================================================
# Main Application Configuration
# ============================================================================
# This section configures the main AutoStash application, which uses the PPCG, PET, and ISL libraries
# to perform code transformations and optimizations.

message(STATUS "Configuring main AutoStash application")

# Build the main application
add_executable(AutoStash main.cpp)
set_target_properties(AutoStash PROPERTIES OUTPUT_NAME "autoStash")
target_link_libraries(AutoStash ppcg pet isl)

# Export include directories so other projects can use these libraries
set(AutoStash_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PPCG_DIR}
    ${ISL_DIR}/include
    ${PET_DIR}/include
    CACHE INTERNAL "AutoStash include directories"
)

# Install AutoStash application
install(TARGETS AutoStash
    RUNTIME DESTINATION bin
)
