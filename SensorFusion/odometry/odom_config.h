#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>
#include "image_tracker.h"
#include "odom_estimator.h"

namespace SensorFusion {

struct OdometryConfig {
    using Ptr = std::shared_ptr<OdometryConfig>;
    OdometryConfig() {
        image_tracker_config.reset(new ImageTrackerConfig);
        odom_estimator_config.reset(new OdomEstimatorConfig);
    }

    OdometryConfig(const std::string &config_path) {
        if (config_path == "") {
            image_tracker_config.reset(new ImageTrackerConfig);
            odom_estimator_config.reset(new OdomEstimatorConfig);
        } else {
            auto config_node = YAML::LoadFile(config_path);
            if (config_node) {
                if (config_node["ImageTracker"])
                    image_tracker_config.reset(
                        new ImageTrackerConfig(config_node["ImageTracker"]));
                else
                    image_tracker_config.reset(new ImageTrackerConfig());

                if (config_node["OdomEstimator"])
                    odom_estimator_config.reset(
                        new OdomEstimatorConfig(config_node["OdomEstimator"]));
                else
                    odom_estimator_config.reset(new OdomEstimatorConfig());
            } else {
                image_tracker_config.reset(new ImageTrackerConfig);
                odom_estimator_config.reset(new OdomEstimatorConfig);
            }
        }
    }

    ImageTrackerConfig::Ptr image_tracker_config;
    OdomEstimatorConfig::Ptr odom_estimator_config;
};
}  // namespace SensorFusion