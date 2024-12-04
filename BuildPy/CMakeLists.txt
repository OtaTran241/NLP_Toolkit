cmake_minimum_required(VERSION 3.15)
project(pybind_Toolkit)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR})

add_library(pybind_Toolkit MODULE pybind_Toolkit.cpp Toolkit.cpp Tokenizer.cpp)
target_link_libraries(pybind_Toolkit PRIVATE pybind11::module)
