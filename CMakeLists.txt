cmake_minimum_required(VERSION 3.18)
project(py_NLP_Toolkit)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR})

add_library(py_NLP_Toolkit MODULE pybind_NLP_Toolkit.cpp Toolkit.cpp Tokenizer.cpp)
target_link_libraries(py_NLP_Toolkit PRIVATE pybind11::module)
