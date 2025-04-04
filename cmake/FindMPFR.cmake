# FindMPFR.cmake
# 查找MPFR库
# 定义:
#  MPFR_FOUND - 找到了MPFR库
#  MPFR_INCLUDE_DIRS - MPFR头文件目录
#  MPFR_LIBRARIES - MPFR库

# 包括有用的查找功能
include(FindPackageHandleStandardArgs)

# 查找库
find_library(MPFR_LIBRARIES
  NAMES mpfr libmpfr
  PATHS
    ${MPFR_ROOT}
    $ENV{MPFR_ROOT}
    $ENV{CONDA_PREFIX}
    /usr
    /usr/local
    /opt/local
  PATH_SUFFIXES lib lib64
)

# 查找头文件
find_path(MPFR_INCLUDE_DIRS
  NAMES mpfr.h
  PATHS
    ${MPFR_ROOT}
    $ENV{MPFR_ROOT}
    $ENV{CONDA_PREFIX}
    /usr
    /usr/local
    /opt/local
  PATH_SUFFIXES include
)

# 处理变量参数
find_package_handle_standard_args(MPFR
  REQUIRED_VARS MPFR_LIBRARIES MPFR_INCLUDE_DIRS
)

# 设置缓存条目
mark_as_advanced(MPFR_INCLUDE_DIRS MPFR_LIBRARIES) 