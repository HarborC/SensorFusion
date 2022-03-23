cmake_minimum_required(VERSION 3.2)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-unused-parameter -Wno-deprecated-declarations")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wno-deprecated-declarations")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-catch-value")
endif()

#find_package(catkin REQUIRED COMPONENTS cpp_common roscpp_serialization roscpp_traits rostime roslz4)
find_package(Boost REQUIRED COMPONENTS date_time filesystem program_options regex)
find_package(BZip2 REQUIRED)
find_library(lz4_LIBRARIES NAMES lz4)

set(_opencv_version 4)
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
  message(STATUS "Did not find OpenCV 4, trying OpenCV 3")
  set(_opencv_version 3)
endif()

find_package(OpenCV ${_opencv_version} REQUIRED
  COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
  CONFIG
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

file(GLOB CONSOLE_BRIDGE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/console_bridge/src/*.cpp")
file(GLOB CPP_COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/cpp_common/src/*.cpp")
file(GLOB ROSCPP_SERIALIZATION_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/roscpp_serialization/src/*.cpp")
file(GLOB ROSTIME_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/rostime/src/*.cpp")

file(GLOB ROSBAG_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ros_comm/tools/rosbag_storage/src/*.cpp")
file(GLOB ROSLZ4_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ros_comm/utilities/roslz4/src/[a-z]*.c")
file(GLOB CV_BRIDGE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/cv_bridge/src/*.cpp") #) "${CMAKE_CURRENT_SOURCE_DIR}/cv_bridge/src/*.hpp")


add_library(rosbag STATIC ${ROSBAG_SRCS} ${ROSTIME_SRCS} ${CPP_COMMON_SRCS} ${ROSCPP_SERIALIZATION_SRCS} ${ROSLZ4_SRCS} ${CONSOLE_BRIDGE_SRCS} ${CV_BRIDGE_SRCS})

target_include_directories(rosbag PUBLIC
    include
    console_bridge/include
    roscpp_core/cpp_common/include
    roscpp_core/rostime/include
    roscpp_core/roscpp_serialization/include
    roscpp_core/roscpp_traits/include
    ros_comm/utilities/roslz4/include
    ros_comm/tools/rosbag_storage/include
    cv_bridge/include
    cv_bridge/src
    pcl_conversions/include)

target_link_libraries(rosbag PUBLIC ${Boost_LIBRARIES} ${BZIP2_LIBRARIES} ${lz4_LIBRARIES} ${OpenCV_LIBRARIES})

