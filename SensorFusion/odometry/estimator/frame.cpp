#include "frame.h"
#include "sensor_flag.h"

namespace SensorFusion {

Frame::Frame() {
    sensorType = SensorFlag::UNKNOWN;
    pre_imu_enabled = false;
    P = Eigen::Vector3d::Zero();
    Q = Eigen::Quaterniond::Identity();
    V = Eigen::Vector3d::Zero();
    bg = Eigen::Vector3d::Zero();
    ba = Eigen::Vector3d::Zero();
}

Frame::Frame(const Frame::Ptr& frame) {
    timeStamp = frame->timeStamp;
    sensorType = frame->sensorType;
    imuIntegrator = frame->imuIntegrator;
    pre_imu_enabled = frame->pre_imu_enabled;
    sensor_id = frame->sensor_id;
    P = frame->P;
    Q = frame->Q;
    V = frame->V;
    bg = frame->bg;
    ba = frame->ba;
    ground_plane_coeff = frame->ground_plane_coeff;
    ExT_ = frame->ExT_;
    P_ = frame->P_;
    Q_ = frame->Q_;
}

}  // namespace SensorFusion