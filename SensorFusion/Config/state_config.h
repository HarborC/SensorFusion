#ifndef _SENSOR_FUSION_FILTER_STATE_CONFIG_H_
#define _SENSOR_FUSION_FILTER_STATE_CONFIG_H_

#include "../common_header.h"

namespace SensorFusion {

struct StateConfig {
    // Static scene duration, in seconds
    double zupt_max_feature_dis = 5;
    double features_rate = 5;
    double static_duration = 5;
    int static_num = (int)(static_duration * features_rate);

    // imu0
    double imu_rate = 100;
    double imu_gyro_noise = 0.004;
    double imu_acc_noise = 0.08;
    double imu_gyro_bias_noise = 0.0002;
    double imu_acc_bias_noise = 0.004;

    Eigen::Matrix3d Ma = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Tg = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d As = Eigen::Matrix3d::Zero();

    // cam0
    bool estimate_td_cam0 = true;
    double td_cam0 = 0.0;
    Eigen::Matrix4d T_imu0_cam0 = Eigen::Matrix4d::Identity();

    void print() {
        std::cout << "State Config : " << std::endl;
        // std::cout << " - sliding_window_size : " << sliding_window_size_ <<
        // std::endl; std::cout << " - compute_raw_odom : " << compute_raw_odom_
        // << std::endl; std::cout << " - enable_plane_update : " <<
        // enable_plane_update_ << std::endl; std::cout << " - enable_gps_update
        // : " << enable_gps_update_ << std::endl; std::cout << " -
        // propagator_type : " << propagator_type_ << std::endl;
    }
};

}  // namespace SensorFusion

#endif