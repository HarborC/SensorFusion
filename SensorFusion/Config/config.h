/*
 * @Author: Jiagang Chen
 * @Date: 2021-08-23 04:22:23
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-04 08:01:27
 * @Description: ...
 * @Reference: ...
 */
#ifndef _SENSOR_FUSION_CONFIG_H_
#define _SENSOR_FUSION_CONFIG_H_

#include "../common_header.h"
#include "filter_fusion_config.h"
#include "sensor_config.h"
#include "state_config.h"
#include "system_initializer_config.h"

namespace SensorFusion {

class SystemConfig {
public:
    typedef std::shared_ptr<SystemConfig> Ptr;
    SystemConfig(const std::string &config_path);
    ~SystemConfig() {}

    void print();

public:
    FilterFusionConfig filter_fusion_config_;
    SystemInitializerConfig system_initializer_config_;
    StateConfig state_config_;
    SensorConfig sensor_config_;

protected:
    bool load_param(const std::string &config_path);
    void load_filter_config_param(const YAML::Node &node);
    void load_initializer_config_param(const YAML::Node &node);
    void load_state_config_param(const YAML::Node &node);
    void load_sensor_config_param(const YAML::Node &node);
};

}  // namespace SensorFusion

#endif