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
