cmake_minimum_required(VERSION 3.14)
project(GML)

# For testing with gdb (Use gdb ./GML and type "r" on Terminal to start)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O0")

# TensorFlow
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
find_package(TensorFlow REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}") # This line causes a compatibility issue of ABI between TensorFlow and YOLO (TensorFlow is compiled with an old version ABI by setting _GLIBCXX_USE_CXX11_ABI=0)
# To prevent the ABI issue, re-build YOLO with -D_GLIBCXX_USE_CXX11_ABI=0
# Reference: http://dev.pawelsz.eu/2017/05/tensorflow-custom-operation-problem.html
#           https://github.com/AlexeyAB/darknet/blob/9c1b9a2cf6363546c152251be578a21f3c3caec6/Makefile#L1

set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )

TensorFlow_REQUIRE_C_LIBRARY()
TensorFlow_REQUIRE_SOURCE()

# RealSense
set(REALSENSE_INCLUDE ~/Documents/librealsense-2.29.0/include)
set(REALSENSE_LIB ~/Documents/librealsense-2.29.0/build/librealsense2.so)

# OpenCV
find_package(OpenCV REQUIRED)

# OpenGL & GLUT
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# pthread
find_package (Threads REQUIRED)

# YOLO
#set(YOLO_INCLUDE ~/Documents/darknet/include) # Added #define OPENCV 1 to yolo_v2_class.hpp and so copy the header file to src folder instead
set(YOLO_LIB /usr/local/cuda-10.0/lib64/libdarknet.so)

# include files
set(INCLUDE_DIR ${REALSENSE_INCLUDE} ${OpenCV_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS})
include_directories("${INCLUDE_DIR}")

add_executable(GML main.cpp yolo_v2_class.hpp session.cpp session.h extern.h extern.cpp maestro.h maestro.cpp modelApp.h)
target_include_directories(GML PRIVATE TensorFlow_DEP)

# Library files
set(LIB_DIR ${REALSENSE_LIB} ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${YOLO_LIB} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(GML PRIVATE TensorFlow_DEP ${LIB_DIR} "stdc++fs")

# Deactivate when using gdb
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Notes for building this program on Linux
# 1) Add #include <experimental/filesystem>
# 2) Add "stdc++fs" to target_link_libraries in CMakeLists.txt

