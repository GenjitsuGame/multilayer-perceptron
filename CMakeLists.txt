cmake_minimum_required(VERSION 3.6)
project(ML1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp mlp.cpp mlp.h)
add_executable(ML1 ${SOURCE_FILES})