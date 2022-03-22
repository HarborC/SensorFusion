cmake_minimum_required(VERSION 3.10...3.18)
project(SensorFusion)

add_definitions(-w)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#############################################
##############    PACKAGE   #################
#############################################

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)

# Boost
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
include_directories(${Boost_INCLUDE_DIRS})
message("BOOST VERSION:" ${Boost_VERSION})

# # Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message("Eigen3 VERSION:" ${EIGEN3_VERSION})

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# GLOG 
find_package(GLOG REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
message("GLOG [OK]:" ${GLOG_INCLUDE_DIRS})

# Ceres
# find_package(Ceres REQUIRED)
# find_package(SuiteSparse REQUIRED)
# include_directories(${CERES_INCLUDE_DIRS})
# include_directories(${SUITESPARSE_INCLUDE_DIRS})
# message("CERES [OK]:" ${CERES_INCLUDE_DIRS})

# # Gtsam
# find_package(GTSAM REQUIRED)
# message("GTSAM [OK]: " ${GTSAM_INCLUDE_DIR})
# include_directories(${GTSAM_INCLUDE_DIR})

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
message("PCL [OK]:" ${PCL_INCLUDE_DIRS})

# TBB
find_package(TBB REQUIRED)
message(STATUS "Found TBB ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} (interface version ${TBB_INTERFACE_VERSION}) headers in: ${TBB_INCLUDE_DIRS}")
if (TBB_INTERFACE_VERSION LESS 11004)
  # enable global_control header for earlier TBB versions (Ubuntu 16.04, 18.04)
  add_definitions(-DTBB_PREVIEW_GLOBAL_CONTROL)
endif()

# OpenCV
# NOTE: not specifying version, since 2, 3 or 4 is fine
find_package(OpenCV REQUIRED) # COMPONENTS core imgproc calib3d highgui)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "Found OpenCV ${OpenCV_VERSION} headers in: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV_LIBS: ${OpenCV_LIBS}")

# fmt
find_package(fmt REQUIRED)
message(STATUS "Found {fmt} ${fmt_VERSION} in: ${fmt_DIR}")

add_subdirectory(thirdparty)

# custom scoped cli11 target
add_library(SensorFusion::cli11 INTERFACE IMPORTED)
set_property(TARGET SensorFusion::cli11 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/CLI11/include)

# custom scoped magic_enum target
add_library(SensorFusion::magic_enum INTERFACE IMPORTED)
set_property(TARGET SensorFusion::magic_enum PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/magic_enum/include)


# Include
include_directories(
  ${PROJECT_SOURCE_DIR}/SensorFusion/
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

file(GLOB SensorFusion_Odometry_Files
  "SensorFusion/odometry/*.h"
  "SensorFusion/odometry/*.cc"
  "SensorFusion/odometry/*.cpp"
  "SensorFusion/odometry/*.hpp"
  "SensorFusion/odometry/camera_model/*.h"
  "SensorFusion/odometry/camera_model/*.cc"
  "SensorFusion/odometry/camera_model/*.cpp"
  "SensorFusion/odometry/camera_model/*.hpp"
  "SensorFusion/odometry/estimator/*.h"
  "SensorFusion/odometry/estimator/*.cc"
  "SensorFusion/odometry/estimator/*.cpp"
  "SensorFusion/odometry/estimator/*.hpp"
  "SensorFusion/odometry/visualizer/*.h"
  "SensorFusion/odometry/visualizer/*.cc"
  "SensorFusion/odometry/visualizer/*.cpp"
  "SensorFusion/odometry/visualizer/*.hpp"
  
  "SensorFusion/odometry/nano_gicp/*.h"
  "SensorFusion/odometry/nano_gicp/*.cc"
  "SensorFusion/odometry/nano_gicp/*.cpp"
  "SensorFusion/odometry/nano_gicp/*.hpp"
  "SensorFusion/odometry/nano_gicp/gicp/*.h"
  "SensorFusion/odometry/nano_gicp/gicp/*.cc"
  "SensorFusion/odometry/nano_gicp/gicp/*.cpp"
  "SensorFusion/odometry/nano_gicp/gicp/*.hpp"
  "SensorFusion/odometry/nano_gicp/impl/*.h"
  "SensorFusion/odometry/nano_gicp/impl/*.cc"
  "SensorFusion/odometry/nano_gicp/impl/*.cpp"
  "SensorFusion/odometry/nano_gicp/impl/*.hpp"
)

file(GLOB SensorFusion_Files
  ${SensorFusion_Odometry_Files}
)

add_library(${PROJECT_NAME} SHARED
  ${SensorFusion_Files}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC ${STD_CXX_FS} ${OpenCV_LIBS} TBB::tbb yaml-cpp ${Boost_LIBRARIES} ${GLOG_LIBRARIES} ${PCL_LIBRARIES} ceres pangolin # ${CERES_LIBRARIES}  
  PRIVATE SensorFusion::magic_enum rosbag apriltag opengv nlohmann::json fmt::fmt Sophus::Sophus cereal::cereal 
)
target_compile_definitions(${PROJECT_NAME} PUBLIC ${SENSOR_FUSION_COMPILE_DEFINITIONS})

#############################################
#############    executable   ###############
#############################################

add_executable(odometry_fusion_kaist app/odometry_fusion_kaist.cpp)
target_link_libraries(odometry_fusion_kaist ${PROJECT_NAME} SensorFusion::cli11)

add_executable(odometry_fusion_nclt app/odometry_fusion_nclt.cpp)
target_link_libraries(odometry_fusion_nclt ${PROJECT_NAME} SensorFusion::cli11)