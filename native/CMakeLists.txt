cmake_minimum_required(VERSION 3.24)
project(git_brew_native LANGUAGES CXX CUDA)

# The shipped list is real SASS for every architecture the toolkit can
# target, plus sm_120 PTX for whatever comes after Blackwell. Plain -real,
# no a/f suffix: the kernel needs nothing newer than lop3 (sm_50).
# Minor-forward cubin compatibility fills the gaps: 80-real runs on sm_87
# (Orin), 100-real on sm_103, 120-real on sm_121 (GB10). Thor is covered by
# neither a neighbouring cubin nor the sm_120 PTX, so it is listed under
# the name its toolkit gives it: sm_110 from CUDA 13.0, sm_101 before.
# CUDA 12.8 is the floor because it is the first to target Blackwell.
#
# The list is chosen here rather than in build.rs because only CMake knows
# which toolkit it found. A plain set() shadows any cached arch list left
# behind by an earlier override build.
option(GIT_BREW_SHIPPED_ARCHS "Build the multi-arch fatbin" OFF)
if(GIT_BREW_SHIPPED_ARCHS)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        set(_thor_arch 110-real)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
        set(_thor_arch 101-real)
    else()
        message(FATAL_ERROR
            "The shipped arch list needs CUDA >= 12.8, found "
            "${CMAKE_CUDA_COMPILER_VERSION}. Set GIT_BREW_CUDA_ARCH to "
            "build for a narrower list on an older toolkit.")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES
        75-real 80-real 86-real 89-real 90-real 100-real ${_thor_arch}
        120-real 120-virtual)
# A standalone tree built to run ctest on a dev machine targets the local
# GPU. GPU-less builds must pass -DCMAKE_CUDA_ARCHITECTURES explicitly;
# native detection needs a device.
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
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
