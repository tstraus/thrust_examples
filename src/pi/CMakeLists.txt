project (pi)

add_executable (${PROJECT_NAME} pi.cpp)

target_link_libraries (${PROJECT_NAME} OpenCL::OpenCL)

install (TARGETS ${PROJECT_NAME} DESTINATION bin)

