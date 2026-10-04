cmake_minimum_required (VERSION 3.10)
project (NeuralNetworkFramework)

set (NeuralNetworkFramework_VERSION_MAJOR 1)
set (NeuralNetworkFramework_VERSION_MINOR 0)

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/main.cpp
	${PROJECT_SOURCE_DIR}/src/ImagesLoader.cpp
	${PROJECT_SOURCE_DIR}/src/ImagesLoader.h
	${PROJECT_SOURCE_DIR}/src/Model.cpp
	${PROJECT_SOURCE_DIR}/src/Model.h
	${PROJECT_SOURCE_DIR}/src/ThreadPool.cpp
	${PROJECT_SOURCE_DIR}/src/ThreadPool.h)

add_executable(NeuralNetworkFramework ${SOURCE_FILES})

target_include_directories(NeuralNetworkFramework
	PRIVATE
		"${PROJECT_BINARY_DIR}"
		"${PROJECT_SOURCE_DIR}/src/")

if(APPLE)
    # 1. MINIMUM BOUND: Allows the binary to run on your local macOS 26 machine
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.6" CACHE STRING "Minimum macOS version allowed" FORCE)
    
    # 2. UPPER BOUND (API ceiling): Points Clang directly to the macOS 27 SDK paths
    #set(CMAKE_OSX_SYSROOT "/path/to/MacOSX27.0.sdk" CACHE PATH "macOS SDK Location" FORCE)
    
    # 3. SAFETY SWITCH: Stops CMake from crashing when it validates your cross-compilation compiler
    #set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
endif()

# ---- tests ----
# gtest is fetched at configure time; the first configure needs network access.
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(NeuralNetworkTests
    ${PROJECT_SOURCE_DIR}/tests/ImageLoaderTests.cpp
    ${PROJECT_SOURCE_DIR}/tests/ModelTests.cpp
    ${PROJECT_SOURCE_DIR}/src/ImagesLoader.cpp
    ${PROJECT_SOURCE_DIR}/src/Model.cpp
    ${PROJECT_SOURCE_DIR}/src/ThreadPool.cpp)

target_include_directories(NeuralNetworkTests
    PRIVATE
        "${PROJECT_SOURCE_DIR}/src/")

target_link_libraries(NeuralNetworkTests PRIVATE GTest::gtest_main)

include(GoogleTest)
# PRE_TEST enumerates tests when ctest runs, not as an Xcode post-build script,
# which fails under Xcode's script sandbox.
gtest_discover_tests(NeuralNetworkTests DISCOVERY_MODE PRE_TEST)
