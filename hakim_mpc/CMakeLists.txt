cmake_minimum_required(VERSION 3.0)
project(hakim_mpc)


if(UNIX)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wfatal-errors -Werror=return-type -g -O0 -no-pie -fno-pie")
    message("Adding g++ specific flags")
endif()

# set the C++ standard to C++ 20
set(CMAKE_CXX_STANDARD 20)
# set the C++ compiler to use O3
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(mpc++ CONFIG REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=0)

set(MPC_LINK_LIB 
    "mpc++" 
    ${OpenMP_CXX_LIBRARIES})




# # Declare a C++ library
include_directories(${mpc++_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} rov_example.cpp)
target_link_libraries(${PROJECT_NAME} mpc++)
