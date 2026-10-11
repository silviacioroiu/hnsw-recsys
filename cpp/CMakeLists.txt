cmake_minimum_required(VERSION 3.16)
project(hnsw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The Python extension in M4 links this library into a shared object.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(hnsw src/hnsw.cpp)
target_include_directories(hnsw PUBLIC include)
target_compile_options(hnsw PRIVATE -Wall -Wextra -Wpedantic)

option(HNSW_BUILD_PYTHON "Build the pybind11 module recsys._hnsw" OFF)
if(HNSW_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)  # use CMake's FindPython, not the removed legacy module
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_hnsw python/bindings.cpp)
  target_link_libraries(_hnsw PRIVATE hnsw)
  # Placed inside the Python package so `import recsys._hnsw` just works.
  set_target_properties(_hnsw PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/recsys)
endif()

# Give extracted files the extraction time, so a changed URL triggers a rebuild.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(hnsw_tests tests/test_hnsw.cpp)
target_link_libraries(hnsw_tests PRIVATE hnsw GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(hnsw_tests)
