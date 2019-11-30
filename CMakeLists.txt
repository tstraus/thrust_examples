cmake_minimum_required (VERSION 3.14.0)

project (compute_examples)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set (Boost_USE_STATIC_LIBS OFF)
#set (Boost_USE_MULTITHREADED ON)
#set (Boost_USE_STATIC_RUNTIME OFF)
#find_package (BoostCompute REQUIRED)

find_package (OpenCL REQUIRED)

include_directories (/usr/include/boost)

add_subdirectory (src)

