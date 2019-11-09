cmake_minimum_required (VERSION 3.14.0)

project (thrust_examples LANGUAGES CUDA CXX)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set (CMAKE_CUDA_STANDARD 11)
    set (CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

add_subdirectory (src)

