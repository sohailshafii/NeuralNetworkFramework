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
	${PROJECT_SOURCE_DIR}/src/Model.h)

add_executable(NeuralNetworkFramework ${SOURCE_FILES})

target_include_directories(NeuralNetworkFramework
	PRIVATE
		"${PROJECT_BINARY_DIR}"
		"${PROJECT_SOURCE_DIR}/src/")


