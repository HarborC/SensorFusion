#pragma once

#include <Eigen/Core>
#include <iostream>
#include <memory>

namespace SensorFusion {

/** \brief imu data struct */
struct ImuData {
    using Ptr = std::shared_ptr<ImuData>;
    double timeStamp;
    Eigen::Vector3d accel, gyro;
    Eigen::Vector3d accel_bias, gyro_bias;
};

}  // namespace SensorFusion