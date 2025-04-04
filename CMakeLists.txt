include(CheckCSourceCompiles)

# 类似check_c_source_compiles，但设置结果为0或1
function(check_c_source_compiles_numeric _prog _var)
  check_c_source_compiles("${_prog}" "${_var}_RESULT")
  if(${_var}_RESULT)
    set(${_var} 1 PARENT_SCOPE)
  else()
    set(${_var} 0 PARENT_SCOPE)
  endif()
endfunction()

# 在CMake文件开头添加以下内容，强制使用系统GCC
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# 获取编译器信息，包括默认include路径
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -E -x c++ -v /dev/null
    ERROR_VARIABLE compiler_info
    OUTPUT_QUIET
)

# 解析输出以获取include路径 - 修复路径过滤
string(REGEX MATCHALL "/usr/include[^\n]+|/usr/lib[^\n]+" include_paths "${compiler_info}")

# 修复__has_feature宏函数定义
add_compile_options(-D"__has_feature(x)=0")

# 单独设置CLANG_PREFIX宏
execute_process(
    COMMAND ${LLVM_CONFIG} --prefix
    OUTPUT_VARIABLE CLANG_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_compile_definitions(CLANG_PREFIX="${CLANG_PREFIX}")


cmake_minimum_required(VERSION 3.10)
project(AutoStash LANGUAGES C CXX)

# 设置C/C++标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# 编译选项
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_OPENCL "Enable OpenCL support" OFF)

# 设置查找模块路径
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# 查找必要的库
# GMP和MPFR库的检测
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

# 在查找GMP之后添加LLVM/Clang检测
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

# 获取LLVM/Clang包含路径和库
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM include目录: ${LLVM_INCLUDE_DIRS}")
message(STATUS "Clang库: ${CLANG_LIBRARIES}")

# 设置Clang编译选项 - 修复定义方式
set(CLANG_CXXFLAGS "-fno-rtti -fPIC")

# 添加系统标准库路径，仅包含有效的目录
foreach(path ${include_paths})
    if(IS_DIRECTORY "${path}")
        include_directories(SYSTEM "${path}")
        message(STATUS "Adding include path: ${path}")
    endif()
endforeach()

# 添加GMP和MPFR头文件目录
include_directories(${GMP_INCLUDE_DIRS} ${MPFR_INCLUDE_DIRS})
include_directories(${CMAKE_CXX_STANDARD_INCLUDE_DIRS})

# 确保包含标准C++头文件路径
include_directories(SYSTEM ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})

# OpenMP和OpenCL的检测
if(USE_OPENMP)
    find_package(OpenMP)
endif()
if(USE_OPENCL)
    find_package(OpenCL)
endif()

# 设置第三方库路径
set(PPCG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ppcg)
set(ISL_DIR ${PPCG_DIR}/isl)
set(PET_DIR ${PPCG_DIR}/pet)

# 创建gitversion.h文件（PPCG需要）
execute_process(
    COMMAND git describe --always 
    WORKING_DIRECTORY ${PPCG_DIR}
    OUTPUT_VARIABLE GIT_HEAD_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(WRITE ${PPCG_DIR}/gitversion.h "#define GIT_HEAD_ID \"${GIT_HEAD_VERSION}\"")

# 创建ISL配置头文件
# 创建编译时需要的目录
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/isl)

# 创建isl_config.h文件
file(WRITE ${CMAKE_BINARY_DIR}/include/isl_config.h "
/* 配置文件，定义了ISL库需要的宏 */
/* 检查编译器是否支持 warn_unused_result 属性 */
#if defined(__GNUC__) && __GNUC__ >= 3
#define WARN_UNUSED __attribute__((__warn_unused_result__))
#else
#define WARN_UNUSED
#endif

/* isl库的属性定义 */
#define __isl_give
#define __isl_take
#define __isl_keep
#define __isl_export
#define __isl_import
#define __isl_null
#define __isl_only
#define __isl_consumed

/* 其他ISL库配置 */
#define USE_GMP_FOR_MP 1
#define GMP_FOR_MP 1
#define HAVE_DECL_FFS 1
#define HAVE_DECL_STRCASECMP 1
#define HAVE_DECL_STRNCASECMP 1
#define HAVE_DECL_SNPRINTF 1

/* 标准头文件检查 */
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_STRINGS_H 1
#define HAVE_INTTYPES_H 1
")

# 创建stdint.h文件
file(WRITE ${CMAKE_BINARY_DIR}/include/isl/stdint.h "#include <stdint.h>")

# 包含编译时生成的头文件目录
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_CXX_STANDARD_INCLUDE_DIRS}) 

# 为ISL和PET库定义编译选项
add_definitions(
    -D_GNU_SOURCE
    -DHAVE_CONFIG_H
    -DGMP_FOR_MP
    -DUSE_GMP_FOR_MP
    -D__STDC_LIMIT_MACROS
)

# 构建ISL库
include_directories(
    ${ISL_DIR}/include
    ${ISL_DIR}
    ${ISL_DIR}/imath
)

# ISL库源文件 - 基于Polly项目的配置
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
    # GMP相关文件
    ${ISL_DIR}/isl_gmp.c
    ${ISL_DIR}/isl_val_gmp.c
    # 模板实现的C文件
    ${ISL_DIR}/set_to_map.c
    ${ISL_DIR}/set_from_map.c
    ${ISL_DIR}/uset_to_umap.c
    ${ISL_DIR}/uset_from_umap.c
)

add_library(isl ${ISL_SOURCES})
target_link_libraries(isl ${GMP_LIBRARIES})
target_compile_definitions(isl PRIVATE
    _GNU_SOURCE
    HAVE_CONFIG_H
    GMP_FOR_MP
    USE_GMP_FOR_MP
    __STDC_LIMIT_MACROS
)

# 尝试为ISL添加调试选项，以便更好地诊断错误
target_compile_options(isl PRIVATE -g)

# 构建PET库 - 根据Makefile.am明确列出需要的源文件
include_directories(
    ${PET_DIR}/include
    ${ISL_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/include/pet
    ${CMAKE_BINARY_DIR}/include/compat
)

# 确保C++标准库路径正确包含
include_directories(SYSTEM ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})

# 完整PET源文件列表（根据Makefile.am）
set(PET_SOURCES
    ${PET_DIR}/aff.c
    ${PET_DIR}/array.c
    ${PET_DIR}/clang.cc       # 新增Clang相关源文件
    ${PET_DIR}/context.c
    ${PET_DIR}/expr.c
    ${PET_DIR}/expr_arg.c
    ${PET_DIR}/expr_plus.cc   # C++源文件
    ${PET_DIR}/filter.c
    ${PET_DIR}/id.cc          # 必须保留的C++源文件
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

# 设置C++源文件编译选项 - 修复源文件属性设置方式
set(CPP_SOURCE_FILES
    ${PET_DIR}/clang.cc
    ${PET_DIR}/expr_plus.cc
    ${PET_DIR}/id.cc
    ${PET_DIR}/inlined_calls.cc
    ${PET_DIR}/inliner.cc
    ${PET_DIR}/killed_locals.cc
    ${PET_DIR}/scan.cc
    ${PET_DIR}/scop_plus.cc
    ${PET_DIR}/substituter.cc
    ${PET_DIR}/version.cc
    ${PET_DIR}/pet.cc
)

# 添加Clang依赖库
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

# 构建pet库
add_library(pet ${PET_SOURCES})
target_link_libraries(pet 
    isl 
    ${GMP_LIBRARIES}
    ${CLANG_REQUIRED_LIBS}
    ${LLVM_LIBRARIES}
)
target_compile_definitions(pet PRIVATE
    _GNU_SOURCE
    HAVE_CONFIG_H
    __STDC_CONSTANT_MACROS
    __STDC_LIMIT_MACROS
    USE_NESTED_ARRAY_SIZE_MODIFIER
    USE_GET_BEGIN_END_LOC
    GETTYPEINFORETURNSTYPEINFO
)
target_include_directories(pet PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
)

# 创建PET配置头文件
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/pet)
file(WRITE ${CMAKE_BINARY_DIR}/include/pet/config.h "
/* 配置文件，定义了PET库需要的宏 */
#define DiagnosticInfo Diagnostic
#define HandleTopLevelDeclReturn bool
#define HandleTopLevelDeclContinue true
#define CREATETARGETINFO_TAKES_SHARED_PTR 1
#define SETINVOCATION_TAKES_SHARED_PTR
#define ADDPATH_TAKES_4_ARGUMENTS
#define ext_implicit_function_decl_c99 ext_implicit_function_decl
#define HAVE_TRANSLATELINECOL 1
#define HAVE_DECL_FFS 1
#define HAVE_DECL_STRCASECMP 1
#define HAVE_DECL_STRNCASECMP 1
#define HAVE_GETRUSAGE 1
#define HAVE_GETTIMEOFDAY 1
#define HAVE_SSIZE_T 1
#define PACKAGE \"pet\"
#define PACKAGE_BUGREPORT \"\"
#define PACKAGE_NAME \"pet\"
#define PACKAGE_STRING \"pet 0.11\"
#define PACKAGE_TARNAME \"pet\"
#define PACKAGE_URL \"\"
#define PACKAGE_VERSION \"0.11\"
#define VERSION \"0.11\"
#define NO_CLANG 1
#define HAVE_STDINT_H 1
#define USE_NESTED_ARRAY_SIZE_MODIFIER 1
#define USE_GET_BEGIN_END_LOC 1
#define getFileLocWithOffset getLocWithOffset
#define GETTYPEINFORETURNSTYPEINFO 1
#define HAVE_BASIC_DIAGNOSTICOPTIONS_H 1
#define HAVE_LEX_HEADERSEARCHOPTIONS_H 1
#define HAVE_CLANG_BASIC_LANGSTANDARD_H 1
#define HAVE_LEX_PREPROCESSOROPTIONS_H 1
#define HAVE_LLVM_OPTION_ARG_H 1
#define CREATEPREPROCESSOR_TAKES_TUKIND 1
#define HAVE_SETMAINFILEID 1
#define SETLANGDEFAULTS_TAKES_5_ARGUMENTS 1
#define SETLANGDEFAULTS CompilerInvocation
#define IK_C Language::C
#define HAVE_BEGIN_END_LOC 1
")

# 构建PPCG
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
    # 从库中移除ppcg.c，避免main函数冲突
    # ${PPCG_DIR}/ppcg.c
    ${PPCG_DIR}/print.c
    ${PPCG_DIR}/util.c
    ${PPCG_DIR}/version.c
    # 添加包含ppcg_options_new_with_defaults实现的文件
    ${CMAKE_CURRENT_SOURCE_DIR}/ppcg_wrapper.c
)

add_library(ppcg ${PPCG_SOURCES})
target_link_libraries(ppcg pet isl ${GMP_LIBRARIES} ${MPFR_LIBRARIES})
if(OpenMP_FOUND)
    target_link_libraries(ppcg OpenMP::OpenMP_C)
endif()
if(OpenCL_FOUND)
    target_link_libraries(ppcg OpenCL::OpenCL)
endif()
target_compile_definitions(ppcg PRIVATE
    _GNU_SOURCE
    HAVE_CONFIG_H
)

# 构建PPCG可执行文件
add_executable(ppcg_exe ${PPCG_DIR}/ppcg.c)
set_target_properties(ppcg_exe PROPERTIES OUTPUT_NAME "ppcg")
target_link_libraries(ppcg_exe ppcg pet isl ${GMP_LIBRARIES} ${MPFR_LIBRARIES})
if(OpenMP_FOUND)
    target_link_libraries(ppcg_exe OpenMP::OpenMP_C)
endif()
if(OpenCL_FOUND)
    target_link_libraries(ppcg_exe OpenCL::OpenCL)
endif()

# 你的主应用程序
add_executable(AutoStash main.cpp)
target_link_libraries(AutoStash ppcg pet isl)

# 导出包含目录，使其他项目可以使用这些库
set(AutoStash_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PPCG_DIR}
    ${ISL_DIR}/include
    ${PET_DIR}/include
    CACHE INTERNAL "AutoStash include directories"
)

# 安装规则
install(TARGETS ppcg pet isl
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES 
    ${PPCG_DIR}/ppcg.h
    ${PPCG_DIR}/ppcg_options.h
    DESTINATION include/ppcg
)
install(DIRECTORY 
    ${ISL_DIR}/include/ 
    DESTINATION include/isl
)
install(DIRECTORY 
    ${PET_DIR}/include/ 
    DESTINATION include/pet
)

# 检查编译器特性
check_c_source_compiles("
int func(void) __attribute__((__warn_unused_result__));
int main(void) { return 0; }
" HAS_ATTRIBUTE_WARN_UNUSED_RESULT)

if(HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
  set(GCC_WARN_UNUSED_RESULT "__attribute__((__warn_unused_result__))")
else()
  set(GCC_WARN_UNUSED_RESULT "")
endif()

check_c_source_compiles("
__attribute__ ((unused)) static void foo(void);
int main(void) { return 0; }
" HAVE___ATTRIBUTE__)

# 检查ffs等函数
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

# 创建Clang兼容性头文件
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/compat)
file(WRITE ${CMAKE_BINARY_DIR}/include/compat/clang_compat.h "
#ifndef CLANG_COMPAT_H
#define CLANG_COMPAT_H

// 定义__has_feature作为宏函数
#ifndef __has_feature
#define __has_feature(x) 0
#endif

#define getFileLocWithOffset getLocWithOffset

// 兼容新版本Clang中已更改的API
#include <clang/Basic/SourceManager.h>
namespace clang {
  // 兼容旧版的getLocation
  inline SourceLocation getLocation(SourceManager &SM, FileID FID, unsigned int line, unsigned int col) {
    return SM.translateLineCol(FID, line, col);
  }
  
  // 添加到SourceManager类
  class SourceManagerEx : public SourceManager {
  public:
    void createMainFileID(const FileEntry *FileEntry) {
      this->setMainFileID(this->createFileID(FileEntry, SourceLocation(), SrcMgr::C_User));
    }
  };
}

// 扩展SourceManager类的createMainFileID方法
namespace clang {
  inline void createMainFileID(SourceManager &SM, const FileEntry *file) {
    SM.setMainFileID(SM.createFileID(file, SourceLocation(), SrcMgr::C_User));
  }
}

#endif // CLANG_COMPAT_H
")

# A4标准LLVM类型映射
file(WRITE ${CMAKE_BINARY_DIR}/include/compat/llvm_compat.h "
#ifndef LLVM_COMPAT_H
#define LLVM_COMPAT_H

#ifdef GETTYPEINFORETURNSTYPEINFO
// 新版LLVM的TypeInfo返回值是一个结构体
#else
// 旧版LLVM的TypeInfo返回值是一个pair
namespace clang {
  struct TypeInfo {
    // 模拟旧版本的first成员
    unsigned Width;
  };
}
#endif

#endif // LLVM_COMPAT_H
")

# 在编译前创建一个自定义的预处理文件
file(WRITE ${CMAKE_BINARY_DIR}/include/pet/custom_stddef.h "
#ifndef _CUSTOM_STDDEF_H
#define _CUSTOM_STDDEF_H
#ifndef __has_feature
#define __has_feature(x) 0
#endif
#include_next <stddef.h>
#endif
")

# 添加到包含路径
include_directories(BEFORE ${CMAKE_BINARY_DIR}/include)
include_directories(BEFORE ${CMAKE_BINARY_DIR}/include/compat)

# 添加Clang目录
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
find_package(Clang REQUIRED CONFIG)
include_directories(${CLANG_INCLUDE_DIRS})

# 然后在pet库目标中添加链接依赖
target_link_libraries(pet isl ${GMP_LIBRARIES} clangAST clangBasic)