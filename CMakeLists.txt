cmake_minimum_required(VERSION 3.10)
project(AutoStash LANGUAGES C CXX)

# Set C/C++ standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# General options
option(USE_MPFR   "Disable MPFR  support" OFF)
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_OPENCL "Enable OpenCL support" ON)
option(BUILD_PET_EXE "Enable Build PET executable" ON)

# Set module search path for some dependencies tools
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Configure required dependencies
include(ConfigDEPS)

# Configure all third-party libraries in ppcg directory
set(PPCG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ppcg)
set(ISL_DIR ${PPCG_DIR}/isl)
include(ConfigISL)
set(PET_DIR ${PPCG_DIR}/pet)
include(ConfigPET)
include(ConfigPPCG)

# Configure AutoStash application
message(STATUS "Configuring main AutoStash application")
set(AUTOSTASH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/main.cpp
)
include_directories(AutoStash PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PPCG_DIR}
    ${ISL_DIR}/include
    ${PET_DIR}/include
)

# Build the main application
add_executable(AutoStash ${AUTOSTASH_SOURCES})
set_target_properties(AutoStash PROPERTIES OUTPUT_NAME "autoStash")
target_link_libraries(AutoStash ppcg)

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
