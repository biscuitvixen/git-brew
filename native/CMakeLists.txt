cmake_minimum_required(VERSION 3.24)
project(git_brew_native LANGUAGES CXX CUDA)

# Plain 121, not 121a: the kernel needs nothing newer than lop3 (sm_50);
# the a-suffix would only gate family-specific features away from a future
# non-GB10 build of the same fatbin.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 121)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# The static library links into a position-independent Rust executable.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(brew_native STATIC backend/cuda/kernel.cu)
target_include_directories(
    brew_native
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# The cmake crate on the Rust side picks the library up from <out>/lib.
install(TARGETS brew_native ARCHIVE DESTINATION lib)

include(CTest)

if(BUILD_TESTING)
    add_executable(test_sha1 tests/test_sha1.cpp)
    target_include_directories(test_sha1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME sha1_fixtures COMMAND test_sha1)

    add_executable(test_encode tests/test_encode.cpp)
    target_include_directories(test_encode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME encode_fixtures COMMAND test_encode)

    add_executable(test_parity tests/test_parity.cu)
    target_include_directories(test_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_parity PRIVATE brew_native)
    add_test(NAME gpu_parity COMMAND test_parity)
endif()
