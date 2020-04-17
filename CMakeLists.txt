# 3.8+ for project(LANGUAGES CUDA)
# 3.9+ for OpenMP::OpenMP_CXX
# 3.10+ findopenmp gained support for language-specific components
# 3.11+ for CMake not to add -fopenmp to the nvcc flags
# 3.13+ for target_link_directories
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(cmm LANGUAGES CUDA VERSION 0.1.0.0)

include(CheckLanguage)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

add_library(cmm INTERFACE)
add_library(cmm::cmm ALIAS cmm)

target_include_directories(cmm INTERFACE include)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(
    cmm
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-G>
  )
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(
    cmm
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
endif()


# "this command should be in the source directory root for CTest to find the test file"
enable_testing() 
add_subdirectory(test)
