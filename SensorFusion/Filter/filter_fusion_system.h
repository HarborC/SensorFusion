#ifndef _SENSOR_FUSION_FILTER_FUSION_SYSTEM_H_
#define _SENSOR_FUSION_FILTER_FUSION_SYSTEM_H_

#include "../Config/config.h"
#include "../Data/database.h"
#include "../Sensors/Imu/imu.h"
#include "../common_header.h"

#include "Propagator/imu_propagator.h"
#include "State/state_block.h"

namespace SensorFusion {

class FilterFusionSystem {
public:
    FilterFusionSystem(const SystemConfig::Ptr& system_config,
                       const DataBase::Ptr& data_base);
    ~FilterFusionSystem(){};

    bool get_measurement();
    bool propagate();
    bool augment_state();
    bool update();
    bool marginalize_oldest_state();

    FilterState::Ptr get_filter_state() { return state_; }

private:
    SystemConfig::Ptr system_config_;
    DataBase::Ptr data_base_;

    ImuPropagator::Ptr imu_propagator_;

    FilterState::Ptr state_;
    size_t cur_frame_id_;
    bool marg_old_state_;

protected:
    std::vector<std::vector<DatasetIO::Measurement::Ptr>> sync_data_;

    // 所有模块

    // std::shared_ptr<TGK::Camera::Camera> camera_;
    // std::unique_ptr<Visualizer> viz_;
    // std::unique_ptr<VisualUpdater> visual_updater_;
    // std::unique_ptr<PlaneUpdater> plane_updater_ = nullptr;
    // std::unique_ptr<GpsUpdater> gps_updater_ = nullptr;
    // std::shared_ptr<TGK::ImageProcessor::FeatureTracker> feature_tracker_;
    // std::shared_ptr<TGK::ImageProcessor::SimFeatureTrakcer>
    // sim_feature_tracker_;

    // static long int kFrameId;

    // // Raw wheel Odometry, just for comparison.
    // Eigen::Matrix3d odom_G_R_O_;
    // Eigen::Vector3d odom_G_p_O_;
    // std::unique_ptr<TGK::WheelProcessor::WheelPropagator> wheel_propagator_;

    // // JUST FOR INITIALIZATION.
    // TGK::BaseType::GpsDataConstPtr latest_gps_data_ = nullptr;
};

}  // namespace SensorFusion

#endif