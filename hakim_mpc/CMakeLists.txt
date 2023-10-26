cmake_minimum_required(VERSION 3.0)
project(hakim_mpc)

# set the C++ standard to C++ 20
set(CMAKE_CXX_STANDARD 20)
# set the C++ compiler to use O3
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(mpc++ CONFIG REQUIRED)

# # Declare a C++ library
include_directories(${mpc++_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} vanderpol_example.cpp)
target_link_libraries(${PROJECT_NAME} mpc++)
