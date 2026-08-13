cmake_minimum_required(VERSION 3.20)

project(vessel_gnc VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)
include(CTest)

# --- Dependencies (pinned tags; fetched at configure time) --------------------
# Eigen comes from its official GitLab home (the GitHub mirror is unreliable).
FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    GIT_SHALLOW TRUE
)

# Eigen registers its own (very large) test suite into CTest when BUILD_TESTING
# is ON. Remember the user's intent, suppress Eigen's tests, then restore.
set(_vessel_gnc_build_testing ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3 pybind11)
set(BUILD_TESTING ${_vessel_gnc_build_testing} CACHE BOOL "" FORCE)

# --- Core library --------------------------------------------------------------
add_library(vessel_gnc STATIC
    src/actuator.cpp
    src/controllers.cpp
    src/dynamics.cpp
    src/integrator.cpp
)
target_include_directories(vessel_gnc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vessel_gnc PUBLIC Eigen3::Eigen)
target_compile_options(vessel_gnc PRIVATE -Wall -Wextra -Wpedantic)

# --- Python binding ------------------------------------------------------------
pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE vessel_gnc)

# --- Tests ---------------------------------------------------------------------
if(BUILD_TESTING)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(test_dynamics tests/test_dynamics.cpp)
    target_link_libraries(test_dynamics PRIVATE vessel_gnc GTest::gtest_main)

    add_executable(test_integrator tests/test_integrator.cpp)
    target_link_libraries(test_integrator PRIVATE vessel_gnc GTest::gtest_main)

    add_executable(test_controllers tests/test_controllers.cpp)
    target_link_libraries(test_controllers PRIVATE vessel_gnc GTest::gtest_main)

    add_executable(test_actuator tests/test_actuator.cpp)
    target_link_libraries(test_actuator PRIVATE vessel_gnc GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(test_dynamics)
    gtest_discover_tests(test_integrator)
    gtest_discover_tests(test_controllers)
    gtest_discover_tests(test_actuator)
endif()

option(VESSEL_GNC_BUILD_BENCHMARKS "Build the C++ micro-benchmark" OFF)
if(VESSEL_GNC_BUILD_BENCHMARKS)
    add_executable(benchmark_core benchmarks/benchmark_core.cpp)
    target_link_libraries(benchmark_core PRIVATE vessel_gnc)
endif()

# --- Install -------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS vessel_gnc ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/vessel_gnc DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# The compiled module lands inside the Python package: this is what scikit-build-core
# wheels expect, and it also makes `cmake --install` into a prefix self-contained.
install(TARGETS _core LIBRARY DESTINATION vessel_gnc)
