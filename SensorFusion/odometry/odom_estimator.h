#pragma once

#include <memory>

#include <sophus/se3.hpp>
#include "image_tracker.h"
#include "imu_data.h"

namespace SensorFusion {

struct State {
    using Ptr = std::shared_ptr<State>;
};

struct OdomEstimatorConfig {
    using Ptr = std::shared_ptr<OdomEstimatorConfig>;
    OdomEstimatorConfig() {}
    OdomEstimatorConfig(const YAML::Node &config_node) {
        if (config_node["odom_type"])
            odom_type = "tight";
        if (config_node["sensor_list"])
            for (const auto &sensor : config_node["sensor_list"])
                sensor_list.push_back(sensor.as<std::string>());
    }

    std::string odom_type = "tight";
    std::vector<std::string> sensor_list;
};

class OdomEstimator {
public:
    using Scalar = float;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SE3 = Sophus::SE3<Scalar>;
    using Quat = Eigen::Quaternion<Scalar>;

public:
    using Ptr = std::shared_ptr<OdomEstimator>;
    explicit OdomEstimator(const OdomEstimatorConfig &config,
                           const Calibration::Ptr &calib)
        : config(config), calib(calib) {}
    ~OdomEstimator() {}

    tbb::concurrent_bounded_queue<ImageTrackerResult::Ptr> vision_data_queue;
    tbb::concurrent_bounded_queue<LidarFeatureResult::Ptr> lidar_data_queue;
    tbb::concurrent_bounded_queue<ImuData::Ptr> imu_data_queue_vision;
    tbb::concurrent_bounded_queue<ImuData::Ptr> imu_data_queue_lidar;
    tbb::concurrent_bounded_queue<State::Ptr> *out_state_queue;

    OdomEstimatorConfig config;
    Calibration::Ptr calib;

    void addIMUToQueue(const ImuData::Ptr &data) {
        imu_data_queue_vision.emplace(data);
        imu_data_queue_lidar.emplace(data);
    }

    void addVisionToQueue(const ImageTrackerResult::Ptr &data) {
        vision_data_queue.push(data);
    }

    ImuData::Ptr popFromImuDataQueue(
        tbb::concurrent_bounded_queue<ImuData::Ptr> &imu_queue) {
        ImuData::Ptr data;
        imu_queue.pop(data);
    }
};

class OdomEstimatorFactory {
public:
    static OdomEstimator::Ptr getOdomEstimator(
        const OdomEstimatorConfig &config, const Calibration::Ptr &calib);
};

}  // namespace SensorFusion