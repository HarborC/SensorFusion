#ifndef _SENSOR_FUSION_SYSTEM_INITIALIZER_CONFIG_H_
#define _SENSOR_FUSION_SYSTEM_INITIALIZER_CONFIG_H_

#include "../common_header.h"

namespace SensorFusion {

enum InitializerType { kInitializerOnlyImu = 0, kInitializerVio = 1 };

struct SystemInitializerConfig {
    int sliding_window_size_ = 10;
    bool compute_raw_odom_ = true;

    bool enable_plane_update_ = true;
    bool enable_gps_update_ = true;

    int initializer_type_ = InitializerType::kInitializerVio;

    void print() {
        std::cout << "System Initializer Config : " << std::endl;
        std::cout << " - sliding_window_size : " << sliding_window_size_
                  << std::endl;
        std::cout << " - compute_raw_odom : " << compute_raw_odom_ << std::endl;
        std::cout << " - enable_plane_update : " << enable_plane_update_
                  << std::endl;
        std::cout << " - enable_gps_update : " << enable_gps_update_
                  << std::endl;
        std::cout << " - initializer_type : " << initializer_type_ << std::endl;
    }
};

}  // namespace SensorFusion

#endif