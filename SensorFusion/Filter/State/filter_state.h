#ifndef _SENSOR_FUSION_FILTER_FILTER_STATE_H_
#define _SENSOR_FUSION_FILTER_FILTER_STATE_H_

#include "../../common_header.h"
#include "state_block.h"

namespace SensorFusion {

struct FilterState {
    typedef std::shared_ptr<FilterState> Ptr;

    double timestamp;

    // Extrinsic.
    Extrinsic::Ptr imu_extrinsic_;
    std::vector<Extrinsic::Ptr> cam_extrinsics_;

    ImuState::Ptr imu_state_;

    // Camera states.
    std::deque<CameraState::Ptr> camera_states_;

    // Covariance.
    Eigen::MatrixXd covariance;

    void Update(const Eigen::VectorXd& delta_x) {
        // update imu state.
        imu_state_->Update(
            delta_x.segment(imu_state_->state_idx, imu_state_->size));

        // update camera state.
        for (auto& cam_state : camera_states_) {
            cam_state->Update(
                delta_x.segment(cam_state->state_idx, cam_state->size));
        }
    }
};

}  // namespace SensorFusion

#endif