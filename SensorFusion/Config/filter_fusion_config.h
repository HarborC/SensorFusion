#ifndef _SENSOR_FUSION_FILTER_FUSION_CONFIG_H_
#define _SENSOR_FUSION_FILTER_FUSION_CONFIG_H_

#include "../common_header.h"

namespace SensorFusion {

enum PropagatorType { kPropagatorImu = 0, kPropagatorWheel = 1 };

struct FilterFusionConfig {
    int sliding_window_size_ = 10;
    bool compute_raw_odom_ = true;

    bool enable_plane_update_ = true;
    bool enable_gps_update_ = true;

    int propagator_type_ = PropagatorType::kPropagatorImu;

    void print() {
        std::cout << "Filter Fusion Config : " << std::endl;
        std::cout << " - sliding_window_size : " << sliding_window_size_
                  << std::endl;
        std::cout << " - compute_raw_odom : " << compute_raw_odom_ << std::endl;
        std::cout << " - enable_plane_update : " << enable_plane_update_
                  << std::endl;
        std::cout << " - enable_gps_update : " << enable_gps_update_
                  << std::endl;
        std::cout << " - propagator_type : " << propagator_type_ << std::endl;
    }
};

}  // namespace SensorFusion

#endif