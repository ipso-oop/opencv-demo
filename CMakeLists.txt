cmake_minimum_required(VERSION 3.20)
project(stereo_cup_demo CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── Library (testable core) ──────────────────────────────────────────────────
add_library(cup_lib
    stereo_cup_volume.cpp
    stereo_cup_volume.hpp
)
target_link_libraries(cup_lib PUBLIC ${OpenCV_LIBS})
target_include_directories(cup_lib PUBLIC ${OpenCV_INCLUDE_DIRS} .)

# ── Demo executable ─────────────────────────────────────────────────────────
add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE cup_lib)

# ── Tests (GoogleTest via FetchContent) ──────────────────────────────────────
option(BUILD_TESTS "Build unit and integration tests" ON)

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # Unit tests (fast, deterministic, no test data needed)
    add_executable(unit_tests
        tests/test_volume.cpp
        tests/test_depth_stats.cpp
        tests/test_geometry.cpp
        tests/test_params.cpp
        tests/test_roi.cpp
    )
    target_link_libraries(unit_tests PRIVATE cup_lib gtest_main)
    include(GoogleTest)
    gtest_discover_tests(unit_tests)

    # Integration tests (need test data in tests/data/)
    add_executable(integration_tests
        tests/test_integration.cpp
    )
    target_link_libraries(integration_tests PRIVATE cup_lib gtest_main)
    target_compile_definitions(integration_tests PRIVATE
        TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    )
    gtest_discover_tests(integration_tests)
endif()
