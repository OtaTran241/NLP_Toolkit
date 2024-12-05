cmake_minimum_required(VERSION 3.18)
project(NLP_Toolkit)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR})

add_library(NLP_Toolkit MODULE pybind_NLP_Toolkit.cpp Toolkit.cpp Tokenizer.cpp)
target_link_libraries(NLP_Toolkit PRIVATE pybind11::module)
