#pragma once

#include <iostream>
#include <memory>
#include <string>

#include <Eigen/Core>
#include "../imu_integrator.h"

namespace SensorFusion {

/** \brief frame structure */
struct Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Frame>;
    Frame();
    Frame(const Frame::Ptr& frame);
    virtual ~Frame() {}
    double timeStamp;
    int sensorType;

    IMUIntegrator imuIntegrator;
    bool pre_imu_enabled;
    std::string sensor_id;
    Eigen::Vector3d P;
    Eigen::Quaterniond Q;
    Eigen::Vector3d V;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    Eigen::VectorXf ground_plane_coeff;
    Eigen::Matrix4d ExT_;  // Transformation from Sensor to Body
    Eigen::Vector3d P_;
    Eigen::Quaterniond Q_;
};

}  // namespace SensorFusion