cmake_minimum_required(VERSION 3.18)
project(NLP_Toolkit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES
    main.cpp
    Tokenizer.cpp
    Toolkit.cpp
    ThreadPool.cpp
    pybind_NLP_Toolkit.cpp
)

set(HEADERS
    Tokenizer.h
    Toolkit.h
    ThreadPool.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11 found")
    target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module)
else()
    message(WARNING "pybind11 not found, skipping Python bindings")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
