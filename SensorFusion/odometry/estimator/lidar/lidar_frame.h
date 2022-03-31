#pragma once

#include "../../utils/lidar_utils.h"
#include "../frame.h"
#include "../sensor_flag.h"

namespace SensorFusion {

/** \brief lidar frame structure */
struct LidarFrame : public Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<LidarFrame>;
    LidarFrame() : Frame() { sensorType = SensorFlag::LIDAR; }
    ~LidarFrame() override = default;

    pcl::PointCloud<PointType>::Ptr laserCloud;
    pcl::PointCloud<PointType>::Ptr filterLaserCloud;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
        filterCovs;
};

}  // namespace SensorFusion