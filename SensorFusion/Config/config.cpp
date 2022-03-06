/*
 * @Author: Jiagang Chen
 * @Date: 2021-08-23 04:22:23
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-04 08:36:24
 * @Description: ...
 * @Reference: ...
 */
#include "config.h"

namespace SensorFusion {

SystemConfig::SystemConfig(const std::string &config_path) {
    load_param(config_path);
}

bool SystemConfig::load_param(const std::string &config_path) {
    auto config = YAML::LoadFile(config_path);
    if (config) {
        // 滤波参数配置
        auto filter_config_node = config["FilterConfig"];
        if (filter_config_node) {
            load_filter_config_param(filter_config_node);
        } else {
            LOG(WARNING) << "filter information is empty!!!";
        }

        // 初始化参数配置
        auto intializer_config_node = config["InitializerConfig"];
        if (intializer_config_node) {
            load_initializer_config_param(intializer_config_node);
        } else {
            LOG(WARNING) << "initializer information is empty!!!";
        }

        // 初始化各类状态量参数配置
        auto state_config_node = config["StateConfig"];
        if (state_config_node) {
            load_state_config_param(state_config_node);
        } else {
            LOG(WARNING) << "state information is empty!!!";
        }

        // 初始化各类状态量参数配置
        auto sensor_config_path_node = config["sensor_config_path"];
        if (sensor_config_path_node) {
            load_sensor_config_param(config);
        } else {
            LOG(WARNING) << "sensor information is empty!!!";
        }

    } else {
        LOG(ERROR) << "config path is invaild!!!";
        return false;
    }

    print();

    return true;
}

void SystemConfig::load_filter_config_param(const YAML::Node &node) {
    auto sliding_window_size_node = node["sliding_window_size"];
    if (sliding_window_size_node) {
        filter_fusion_config_.sliding_window_size_ =
            sliding_window_size_node.as<int>();
    }

    auto propagator_type_node = node["propagator_type"];
    if (propagator_type_node) {
        filter_fusion_config_.propagator_type_ = propagator_type_node.as<int>();
    }
}

void SystemConfig::load_initializer_config_param(const YAML::Node &node) {
    auto sliding_window_size_node = node["sliding_window_size"];
    if (sliding_window_size_node) {
        system_initializer_config_.sliding_window_size_ =
            sliding_window_size_node.as<int>();
    }

    auto initializer_type_node = node["initializer_type"];
    if (initializer_type_node) {
        system_initializer_config_.initializer_type_ =
            initializer_type_node.as<int>();
    }
}

void SystemConfig::load_state_config_param(const YAML::Node &node) {}

void SystemConfig::load_sensor_config_param(const YAML::Node &node) {
    sensor_config_.sensor_config_path_ =
        node["sensor_config_path"].as<std::string>();
    sensor_config_.sensor_config_node_ =
        YAML::LoadFile(sensor_config_.sensor_config_path_);

    sensor_config_.auto_load();
}

void SystemConfig::print() {
    std::cout << std::endl << "System Config : " << std::endl << std::endl;

    filter_fusion_config_.print();
    system_initializer_config_.print();
    state_config_.print();
    sensor_config_.print();
}

}  // namespace SensorFusion
