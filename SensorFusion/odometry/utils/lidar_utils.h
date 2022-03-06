#pragma once

// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace SensorFusion {

using PointType = pcl::PointXYZINormal;
typedef pcl::PointCloud<PointType> PointCloudType;

enum LidarType {
    VELODYNE = 0,
    OUSTER = 1,
    LIVOX_HORIZON = 2,
    LIVOX_MID = 3,
    LIVOX_AVIA = 4
};

}  // namespace SensorFusion

namespace PointVelodyne {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace PointVelodyne
POINT_CLOUD_REGISTER_POINT_STRUCT(PointVelodyne::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(
                                      float, intensity,
                                      intensity)(float, time, time)(uint16_t,
                                                                    ring, ring))

namespace PointOuster {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace PointOuster
POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointOuster::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)(std::uint16_t, reflectivity, reflectivity)(
        std::uint8_t, ring, ring)(std::uint16_t, ambient,
                                  ambient)(std::uint32_t, range, range))