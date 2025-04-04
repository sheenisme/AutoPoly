# FindGMP.cmake
# 查找GMP库
# 定义:
#  GMP_FOUND - 找到了GMP库
#  GMP_INCLUDE_DIRS - GMP头文件目录
#  GMP_LIBRARIES - GMP库

# 包括有用的查找功能
include(FindPackageHandleStandardArgs)

# 查找库
find_library(GMP_LIBRARIES
  NAMES gmp libgmp
  PATHS
    ${GMP_ROOT}
    $ENV{GMP_ROOT}
    $ENV{CONDA_PREFIX}
    /usr
    /usr/local
    /opt/local
  PATH_SUFFIXES lib lib64
)

# 查找头文件
find_path(GMP_INCLUDE_DIRS
  NAMES gmp.h
  PATHS
    ${GMP_ROOT}
    $ENV{GMP_ROOT}
    $ENV{CONDA_PREFIX}
    /usr
    /usr/local
    /opt/local
  PATH_SUFFIXES include
)

# 处理变量参数
find_package_handle_standard_args(GMP
  REQUIRED_VARS GMP_LIBRARIES GMP_INCLUDE_DIRS
)

# 设置缓存条目
mark_as_advanced(GMP_INCLUDE_DIRS GMP_LIBRARIES) 