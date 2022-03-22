cmake_minimum_required(VERSION 2.8)
project(visualizer)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

add_definitions(-w)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -fPIC")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  add_definitions(-DCOMPILEDWITHC11)
  message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  add_definitions(-DCOMPILEDWITHC0X)
  message(STATUS "Using flag -std=c++0x.")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


#############################################
##############    PACKAGE   #################
#############################################

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
MESSAGE("OPENCV VERSION:" ${OpenCV_VERSION})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Boost
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
include_directories(${Boost_INCLUDE_DIRS})
message("BOOST VERSION:" ${Boost_VERSION})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Include
include_directories(
  ${PROJECT_SOURCE_DIR}/Visualizer/
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

file(GLOB pangolin_visualization_Files
  "Visualizer/*.h"
  "Visualizer/*.cc"
  "Visualizer/*.cpp"
  "Visualizer/*.hpp"
)

add_library(${PROJECT_NAME} SHARED
  ${pangolin_visualization_Files}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Boost_LIBRARIES}
  ${Pangolin_LIBRARIES}
)

#############################################
#############    executable   ###############
#############################################

# add_executable(gps_visual_align test/gps_visual_align.cc)
# target_link_libraries(gps_visual_align ${PROJECT_NAME})

add_executable(test_visualizer test/test_visualizer.cc)
target_link_libraries(test_visualizer ${PROJECT_NAME})
