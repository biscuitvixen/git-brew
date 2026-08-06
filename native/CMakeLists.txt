cmake_minimum_required(VERSION 3.24)
project(git_brew_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The static library links into a position-independent Rust executable.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

if(BUILD_TESTING)
    add_executable(test_sha1 tests/test_sha1.cpp)
    target_include_directories(test_sha1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME sha1_fixtures COMMAND test_sha1)
endif()
