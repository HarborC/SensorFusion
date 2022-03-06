cmake_minimum_required(VERSION 3.10...3.18)
project(SensorFusion)

add_definitions(-w)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

if(NOT EIGEN_ROOT)
  set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/basalt-headers/thirdparty/eigen")
endif()

string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWERCASE)
find_program(DPKG_PROGRAM dpkg DOC "dpkg program of Debian-based systems")
if(DPKG_PROGRAM)
  execute_process(
    COMMAND ${DPKG_PROGRAM} --print-architecture
    OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif(DPKG_PROGRAM)


find_program(LSB_RELEASE_PROGRAM lsb_release DOC "lsb_release program of Debian-based systems")
if(LSB_RELEASE_PROGRAM)
  execute_process(COMMAND ${LSB_RELEASE_PROGRAM} -rs
    OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(${LSB_RELEASE_ID_SHORT} EQUAL "20.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.71.0, libboost-date-time1.71.0, libboost-program-options1.71.0, libboost-regex1.71.0, libopencv-dev, libglew2.1, libjpeg8, libpng16-16, librealsense2, librealsense2-dkms, librealsense2-gl, librealsense2-utils")

  elseif(${LSB_RELEASE_ID_SHORT} EQUAL "18.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.65.1, libboost-date-time1.65.1, libboost-program-options1.65.1, libboost-regex1.65.1, libopencv-dev, libglew2.0, libjpeg8, libpng16-16, librealsense2, librealsense2-dkms, librealsense2-gl, librealsense2-utils")

  elseif(${LSB_RELEASE_ID_SHORT} EQUAL "16.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.58.0, libboost-date-time1.58.0, libboost-program-options1.58.0, libboost-regex1.58.0, libopencv-dev, libglew1.13, libjpeg8, libpng12-0, libstdc++6, librealsense2, librealsense2-dkms, librealsense2-gl, librealsense2-utils")
  endif()

endif(LSB_RELEASE_PROGRAM)

string(TIMESTAMP PROJECT_VERSION_REVISION "%Y%m%d%H%M")

# Configure CCache if available
if (NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  else()
    message(STATUS "Dind't find ccache")
  endif()
else()
  message(STATUS "Compiler launcher already set. Not configuring ccache.")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release)
endif()

if(NOT CXX_MARCH)
  set(CXX_MARCH native)
endif()

IF(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(SENSOR_FUSION_MARCH_FLAGS "-march=${CXX_MARCH}")
ELSE()
  message(STATUS "Running on Apple ${CMAKE_SYSTEM_PROCESSOR}. Disabled -march flag.")
ENDIF()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Flags used for CHECK_CXX_SOURCE_COMPILES
set(CMAKE_REQUIRED_FLAGS "-Wno-error")


# save flags passed by user
set(SENSOR_FUSION_PASSED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")          # cmake default: "-g"
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")  # cmake default: "-O2 -g -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")                                      # cmake default: "-O3 -DNDEBUG"
set(CMAKE_CXX_FLAGS_CIDEBUG  "-O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")          # CI version with no debug symbols
set(CMAKE_CXX_FLAGS_CIRELWITHDEBINFO "-O3 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")  # CI version with no debug symbols

# base set of compile flags
# set(SENSOR_FUSION_CXX_FLAGS "-Wall -Wextra -Werror -Wno-error=unused-parameter -ftemplate-backtrace-limit=0")
set(SENSOR_FUSION_CXX_FLAGS "-Wall -Wextra -Wno-error=unused-parameter -ftemplate-backtrace-limit=0")

# clang-specific compile flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(SENSOR_FUSION_CXX_FLAGS "${SENSOR_FUSION_CXX_FLAGS} -Wno-exceptions -fcolor-diagnostics -frelaxed-template-template-args -Wno-error=deprecated-declarations")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
      # These are disabled to avoid lot's of warnings in Eigen code with clang 10
      set(SENSOR_FUSION_CXX_FLAGS "${SENSOR_FUSION_CXX_FLAGS} -Wno-misleading-indentation -Wno-error=deprecated-copy")
  endif()

  #   - Added TBB_USE_GLIBCXX_VERSION macro to specify the version of GNU
  #     libstdc++ when it cannot be properly recognized, e.g. when used
  #     with Clang on Linux* OS. Adopted from https://github.com/wjakob/tbb
  if(NOT TBB_USE_GLIBCXX_VERSION AND UNIX AND NOT APPLE)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    endif()
    add_definitions(-DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
  endif()
else()
  set(SENSOR_FUSION_CXX_FLAGS "${SENSOR_FUSION_CXX_FLAGS} -Wno-error=maybe-uninitialized -Wno-error=implicit-fallthrough")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
      # These are disabled to avoid lot's of warnings in Eigen code with gcc-9
      set(SENSOR_FUSION_CXX_FLAGS "${SENSOR_FUSION_CXX_FLAGS} -Wno-error=deprecated-copy")
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
    # These are disabled due to warnings in pangolin
    #set(SENSOR_FUSION_CXX_FLAGS "${SENSOR_FUSION_CXX_FLAGS} -Wno-error=parentheses")
    set(SENSOR_FUSION_CXX_FLAGS "${SENSOR_FUSION_CXX_FLAGS} -Wno-parentheses")
  endif()
endif()


# Set platform / compiler specific compile flags and checks
if(APPLE)
  # Need to investigate how to reliably detect and use OpenMP on macOS...
#  set(USE_OPENMP_DEFAULT OFF)

  # Among others, setting CMAKE_FIND_FRAMEWORK to LAST fixed issues
  # with installed Mono that contains old headers (libpng, ...).
  # See: https://github.com/openMVG/openMVG/issues/1349#issuecomment-401492811
  set(CMAKE_FIND_FRAMEWORK LAST)

  if(CMAKE_SYSTEM_VERSION VERSION_LESS 19.0.0)
      # use brewed llvm's libc++
      include_directories("/usr/local/opt/llvm/include/c++/v1")
      link_directories("/usr/local/opt/llvm/lib")
      add_compile_options("-nostdinc++")
      #set(STD_CXX_FS c++fs)

      # Workaround for cmake not to filter the manually added standard include path
      # See: https://gitlab.kitware.com/cmake/cmake/issues/19227#note_669894
      list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/usr/local/opt/llvm/include/c++/v1")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Detected macOS with non-Apple clang")

  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Detected macOS with Apple clang")
    # Apple clang on macOS < 10.14 Mojave is too old
    if(CMAKE_SYSTEM_VERSION VERSION_LESS 18.0.0)
      message(WARNING "Detected Darwin version ${CMAKE_SYSTEM_VERSION}, which is earlier than macos 10.14 Mojave. Apple clang is too old and not supported. Use clang from homebrew.")
    endif()

  else()
    message(WARNING "Detected macOS with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()

elseif(UNIX)
#  set(USE_OPENMP_DEFAULT ON)

  # assume libstdc++
  set(STD_CXX_FS stdc++fs)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Detected Linux with clang.")

    # Note: Whatever the issue was, this seems to not be an issue any more, at least with more recent clang and Ubuntu versions (tested clang-12 and Ubuntu 18.04)
    #message(WARNING "Clang on Linux is currently not fully supported. You'll likely need to get a recent version of TBB.")
    #set(LINUX_CLANG 1)

  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Detected Linux with gcc.")

  else()
    message(WARNING "Detected Linux with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()

else()
  message(WARNING "Only Linux and macOS are currently supported")
endif()


# OpenMP option and compile flags
#
# Note: OpenMP and TBB don't mix well, so we disable Eigen's parallelization.
# It's trying to paralellize matrix products during SC, which we run in a parallel_reduce using TBB.
# Turns out using OpenMP can slow down the computby factor 10-100x! So for now we discable it completely.
# One place where Eigen's parallelization could still have been useful is the CG solver in the mapper.
# We could in the future investiagte other implementations (paralellized with TBB) or selectively enabling
# Eigen's parallelization just for CG, setting number of threads to 1 everywhere else.
# Another way to ensure Eigen doesn't use OpenMP regardless of how it was built is setting the environment
# variable OMP_NUM_THREADS=1 beofre running the application.
#
# See: https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
#
# If we enable BLAS / LAPACK either directly or via thirdparty libs like ceres,
# make sure to disable OpenMP for the linked BLAS library. In particular on Ubuntu it seems OpenBLAS is often installed,
# and it can have similar issues in multithreaded applications if it's own parallelization with OpenMP is enabled.
# You can set the environment varaibles OPENBLAS_NUM_THREADS=1 or OMP_NUM_THREADS=1. This is also mentioned in the ceres
# installation documentation.
#
# See also: https://github.com/xianyi/OpenBLAS/wiki/faq#multi-threaded
#
# Set EIGEN_DONT_PARALLELIZE to be sure it doesn't use openmp,
# just in case some dependency enables openmp without us noticing.
set(SENSOR_FUSION_CXX_FLAGS "${SENSOR_FUSION_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")

#option(USE_OPENMP "Use OpenMP (e.g. for parallel computation in Eigen)" ${USE_OPENMP_DEFAULT})
#if(USE_OPENMP)
#  message(STATUS "OpenMP Enabled")
#  set(SENSOR_FUSION_CXX_FLAGS "${SENSOR_FUSION_CXX_FLAGS} -fopenmp")
#else()
#  message(STATUS "OpenMP Disabled")
#endif()

# Make specialization for double / float optional.
# Disabling them can be useful for better compile times during development.
option(SENSOR_FUSION_INSTANTIATIONS_DOUBLE "Instatiate templates for Scalar=double." ON)
option(SENSOR_FUSION_INSTANTIATIONS_FLOAT "Instatiate templates for Scalar=float." ON)

if(SENSOR_FUSION_INSTANTIATIONS_DOUBLE)
  list(APPEND SENSOR_FUSION_COMPILE_DEFINITIONS SENSOR_FUSION_INSTANTIATIONS_DOUBLE)
endif()
if(SENSOR_FUSION_INSTANTIATIONS_FLOAT)
  list(APPEND SENSOR_FUSION_COMPILE_DEFINITIONS SENSOR_FUSION_INSTANTIATIONS_FLOAT)
endif()


# setup combined compiler flags
set(CMAKE_CXX_FLAGS "${SENSOR_FUSION_CXX_FLAGS} ${SENSOR_FUSION_MARCH_FLAGS} ${SENSOR_FUSION_PASSED_CXX_FLAGS}")

#############################################
##############    PACKAGE   #################
#############################################

# Boost
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
include_directories(${Boost_INCLUDE_DIRS})
message("BOOST VERSION:" ${Boost_VERSION})

# Eigen
set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.4.0 EXACT REQUIRED MODULE)
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")
if(NOT EIGEN3_INCLUDE_DIR MATCHES "^${EIGEN_ROOT}")
  message(WARNING "Found Eigen headers are outside of specified EIGEN_ROOT '${EIGEN_ROOT}'")
endif()

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# GLOG 
find_package(GLOG REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
message("GLOG [OK]:" ${GLOG_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
message("CERES [OK]:" ${CERES_INCLUDE_DIRS})

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
message(STATUS "Found OpenCV ${OpenCV_VERSION} headers in: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV_LIBS: ${OpenCV_LIBS}")

# Add our own custom scoped opencv target since none is provided by OpenCV itself
add_library(SensorFusion::opencv INTERFACE IMPORTED)
set_property(TARGET SensorFusion::opencv PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
set_property(TARGET SensorFusion::opencv PROPERTY INTERFACE_LINK_LIBRARIES ${OpenCV_LIBS})

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

# #############################################
# #########    ThirdParty PACKAGE  ############
# #############################################

# set(DatasetIO_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rdParty/DatasetIO)
# set(DatasetIO_LIBRARIES ${PROJECT_SOURCE_DIR}/3rdParty/DatasetIO/lib/libDatasetIO.so)
# include_directories(${DatasetIO_INCLUDE_DIR})


# Include
include_directories(
  ${PROJECT_SOURCE_DIR}/SensorFusion/
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

file(GLOB SensorFusion_Filter_Files
  "SensorFusion/Filter/*.h"
  "SensorFusion/Filter/*.cc"
  "SensorFusion/Filter/*.cpp"
  "SensorFusion/Filter/*.hpp"
  "SensorFusion/Filter/Propagator/*.h"
  "SensorFusion/Filter/Propagator/*.cc"
  "SensorFusion/Filter/Propagator/*.cpp"
  "SensorFusion/Filter/Propagator/*.hpp"
  "SensorFusion/Filter/State/*.h"
  "SensorFusion/Filter/State/*.cc"
  "SensorFusion/Filter/State/*.cpp"
  "SensorFusion/Filter/State/*.hpp"
  "SensorFusion/Filter/Utils/*.h"
  "SensorFusion/Filter/Utils/*.cc"
  "SensorFusion/Filter/Utils/*.cpp"
  "SensorFusion/Filter/Utils/*.hpp"
)

file(GLOB SensorFusion_Config_Files
  "SensorFusion/Config/*.h"
  "SensorFusion/Config/*.cc"
  "SensorFusion/Config/*.cpp"
  "SensorFusion/Config/*.hpp"
)

file(GLOB SensorFusion_Initializer_Files
  "SensorFusion/Initializer/*.h"
  "SensorFusion/Initializer/*.cc"
  "SensorFusion/Initializer/*.cpp"
  "SensorFusion/Initializer/*.hpp"
  "SensorFusion/Initializer/Vio/*.h"
  "SensorFusion/Initializer/Vio/*.cc"
  "SensorFusion/Initializer/Vio/*.cpp"
  "SensorFusion/Initializer/Vio/*.hpp"
)

file(GLOB SensorFusion_Sensors_Files
  "SensorFusion/Sensors/*.h"
  "SensorFusion/Sensors/*.cc"
  "SensorFusion/Sensors/*.cpp"
  "SensorFusion/Sensors/*.hpp"
  "SensorFusion/Sensors/Imu/*.h"
  "SensorFusion/Sensors/Imu/*.cc"
  "SensorFusion/Sensors/Imu/*.cpp"
  "SensorFusion/Sensors/Imu/*.hpp"
  "SensorFusion/Sensors/Camera/*.h"
  "SensorFusion/Sensors/Camera/*.cc"
  "SensorFusion/Sensors/Camera/*.cpp"
  "SensorFusion/Sensors/Camera/*.hpp"
  "SensorFusion/Sensors/Camera/CameraModel/*.h"
  "SensorFusion/Sensors/Camera/CameraModel/*.cc"
  "SensorFusion/Sensors/Camera/CameraModel/*.cpp"
  "SensorFusion/Sensors/Camera/CameraModel/*.hpp"
  "SensorFusion/Sensors/Camera/Tracker/*.h"
  "SensorFusion/Sensors/Camera/Tracker/*.cc"
  "SensorFusion/Sensors/Camera/Tracker/*.cpp"
  "SensorFusion/Sensors/Camera/Tracker/*.hpp"
  "SensorFusion/Sensors/Camera/Processor/*.h"
  "SensorFusion/Sensors/Camera/Processor/*.cc"
  "SensorFusion/Sensors/Camera/Processor/*.cpp"
  "SensorFusion/Sensors/Camera/Processor/*.hpp"
)

file(GLOB SensorFusion_Data_Files
  "SensorFusion/Data/*.h"
  "SensorFusion/Data/*.cc"
  "SensorFusion/Data/*.cpp"
  "SensorFusion/Data/*.hpp"
)

file(GLOB SensorFusion_Basalt_Files
  "SensorFusion/basalt/calibration/*.cpp"
  "SensorFusion/basalt/io/*.cpp"
  "SensorFusion/basalt/optical_flow/*.cpp"
  "SensorFusion/basalt/utils/*.cpp"
  "SensorFusion/basalt/vi_estimator/*.cpp"
)

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
)

file(GLOB SensorFusion_Files
  # "SensorFusion/*.h"
  # "SensorFusion/*.cc"
  # "SensorFusion/*.cpp"
  # "SensorFusion/*.hpp"
  # ${SensorFusion_Filter_Files}
  # ${SensorFusion_Sensors_Files}
  # ${SensorFusion_Initializer_Files}
  # ${SensorFusion_Data_Files}
  # ${SensorFusion_Config_Files}
  # ${SensorFusion_Basalt_Files}
  ${SensorFusion_Odometry_Files}
)

add_library(${PROJECT_NAME} SHARED
  ${SensorFusion_Files}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC ${STD_CXX_FS} SensorFusion::opencv basalt::basalt-headers TBB::tbb yaml-cpp ${Boost_LIBRARIES} ${GLOG_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} 
  PRIVATE SensorFusion::magic_enum rosbag apriltag opengv nlohmann::json fmt::fmt 
)
target_compile_definitions(${PROJECT_NAME} PUBLIC ${SENSOR_FUSION_COMPILE_DEFINITIONS})

# # 3rdParty
# target_link_libraries(${PROJECT_NAME}
#   ${DatasetIO_LIBRARIES}  
# )

#############################################
#############    executable   ###############
#############################################

# add_executable(slam_test app/slam_test.cc)
# target_link_libraries(slam_test ${PROJECT_NAME})

# add_executable(imu_propagator_test app/imu_propagator_test.cc)
# target_link_libraries(imu_propagator_test ${PROJECT_NAME})

# add_executable(nclt_dataio_test app/nclt_dataio_test.cc)
# target_link_libraries(nclt_dataio_test ${PROJECT_NAME})

# add_executable(run_slam app/run_slam.cc)
# target_link_libraries(run_slam ${PROJECT_NAME})

add_executable(odometry_fusion app/odometry_fusion.cpp)
target_link_libraries(odometry_fusion ${PROJECT_NAME} pangolin SensorFusion::cli11)
