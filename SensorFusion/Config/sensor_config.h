/*
 * @Author: Jiagang Chen
 * @Date: 2021-11-04 07:57:43
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-04 08:34:13
 * @Description: ...
 * @Reference: ...
 */
#ifndef _SENSOR_FUSION_SENSOR_CONFIG_H_
#define _SENSOR_FUSION_SENSOR_CONFIG_H_

#include "../Sensors/Camera/CameraModel/Camera.h"
#include "../Sensors/Camera/CameraModel/CameraFactory.h"
#include "../common_header.h"

namespace SensorFusion {

struct SensorConfig {
    std::string sensor_config_path_;
    YAML::Node sensor_config_node_;

    int num_of_imu_ = 0;
    int num_of_cam_ = 0;
    int num_of_lidar_ = 0;
    int num_of_gnss_ = 0;
    int num_of_wheel_ = 0;

    std::vector<std::string> camera_models_path_;
    std::vector<camodocal::CameraPtr> camera_models_;

    void auto_load() {
        if (sensor_config_node_["imu"])
            num_of_imu_ = sensor_config_node_["imu"].as<int>();
        if (sensor_config_node_["num_of_cam"])
            num_of_cam_ = sensor_config_node_["num_of_cam"].as<int>();
        if (sensor_config_node_["num_of_lidar"])
            num_of_lidar_ = sensor_config_node_["num_of_lidar"].as<int>();
        if (sensor_config_node_["num_of_gnss"])
            num_of_gnss_ = sensor_config_node_["num_of_gnss"].as<int>();
        if (sensor_config_node_["num_of_wheel"])
            num_of_wheel_ = sensor_config_node_["num_of_wheel"].as<int>();

        for (int i = 0; i < num_of_cam_; i++) {
            std::string calibration_cam_path =
                sensor_config_node_[("cam" + std::to_string(i) + "_calib")
                                        .c_str()]
                    .as<std::string>();
            camera_models_path_.push_back(calibration_cam_path);

            camera_models_.push_back(
                camodocal::CameraFactory::instance()
                    ->generateCameraFromYamlFile(calibration_cam_path));
        }
    }

    void print() {
        std::cout << "Sensor Config : " << std::endl;
        std::cout << " - num_of_imu : " << num_of_imu_ << std::endl;
        std::cout << " - num_of_cam : " << num_of_cam_ << std::endl;
        std::cout << " - num_of_lidar : " << num_of_lidar_ << std::endl;
        std::cout << " - num_of_gnss : " << num_of_gnss_ << std::endl;
        std::cout << " - num_of_wheel : " << num_of_wheel_ << std::endl;

        std::cout << " - camera_models_path : " << std::endl;
        for (size_t i = 0; i < camera_models_path_.size(); i++) {
            std::cout << "     " << camera_models_path_[i] << std::endl;
        }
    }
};

}  // namespace SensorFusion

#endif