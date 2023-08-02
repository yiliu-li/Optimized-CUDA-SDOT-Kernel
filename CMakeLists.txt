
# Check cmake version.
cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)

# Routine name
set(ROUTINE1 sdot)

# Project specification.
project(sdot_kernel LANGUAGES C CXX CUDA)

include(GNUInstallDirs)


# Build mode
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# Building flags
# Global CXX/CUDA flags

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Debug options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -g -lineinfo")


# Target
include(../../cmake/cublas_example.cmake)

include_directories("${CMAKE_SOURCE_DIR}/../../utils")

add_cublas_example(cublas_examples "cuda_${ROUTINE1}" cuda_${ROUTINE1}_kernel.cu)


# By default put binaries in build/bin (pre-install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Installation directories
set(CUBLAS_EXAMPLES_BINARY_INSTALL_DIR "cublas_examples/bin")

# Install examples
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF()
