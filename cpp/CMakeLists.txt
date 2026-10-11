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
