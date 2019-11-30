project (sum)

add_executable (${PROJECT_NAME} sum.cpp)

target_link_libraries (${PROJECT_NAME} OpenCL::OpenCL)

install (TARGETS ${PROJECT_NAME} DESTINATION bin)

