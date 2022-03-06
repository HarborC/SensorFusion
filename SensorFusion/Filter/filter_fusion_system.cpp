#include "filter_fusion_system.h"

namespace SensorFusion {

FilterFusionSystem::FilterFusionSystem(const SystemConfig::Ptr& system_config,
                                       const DataBase::Ptr& data_base)
    : system_config_(system_config), data_base_(data_base), cur_frame_id_(0) {
    imu_propagator_ = std::make_shared<ImuPropagator>();
    state_ = std::make_shared<FilterState>();

    // camera_ = std::make_shared<TGK::Camera::PinholeRadTanCamera>(
    //     param_.cam_intrinsic.width, param_.cam_intrinsic.height,
    //     param_.cam_intrinsic.fx, param_.cam_intrinsic.fy,
    //     param_.cam_intrinsic.cx, param_.cam_intrinsic.cy,
    //     param_.cam_intrinsic.k1, param_.cam_intrinsic.k2,
    //     param_.cam_intrinsic.p1, param_.cam_intrinsic.p2,
    //     param_.cam_intrinsic.k3);

    // const auto triangulator = std::make_shared<TGK::Geometry::Triangulator>(
    //     param_.tri_config, camera_);

    // // Create feature tracker.
    // Eigen::Matrix<double, 8, 1> cam_intrin;
    // cam_intrin << param_.cam_intrinsic.fx, param_.cam_intrinsic.fy,
    //     param_.cam_intrinsic.cx, param_.cam_intrinsic.cy,
    //     param_.cam_intrinsic.k1, param_.cam_intrinsic.k2,
    //     param_.cam_intrinsic.p1, param_.cam_intrinsic.p2;

    // feature_tracker_ =
    // std::make_shared<TGK::ImageProcessor::KLTFeatureTracker>(
    //     param_.tracker_config);
    // sim_feature_tracker_ =
    //     std::make_shared<TGK::ImageProcessor::SimFeatureTrakcer>();

    // visual_updater_ = std::make_unique<VisualUpdater>(
    //     param_.visual_updater_config, camera_, feature_tracker_,
    //     triangulator);

    // if (config_.enable_gps_update) {
    //     gps_updater_ =
    //     std::make_unique<GpsUpdater>(param_.extrinsic.C_p_Gps);
    // }
}

bool FilterFusionSystem::get_measurement() {
    if (system_config_->filter_fusion_config_.propagator_type_ == 0) {
        if (data_base_->get_sync_data(sync_data_)) {
            return true;
        }
    }

    return false;
}

bool FilterFusionSystem::propagate() {
    if (system_config_->filter_fusion_config_.propagator_type_ == 0) {
        for (size_t d_i = 0; d_i < sync_data_.size(); d_i++) {
            if (sync_data_[d_i][0]->type == DatasetIO::MeasureType::kImu) {
                auto& imu_data_segment = sync_data_[d_i];

                // propagate state.
                for (size_t i = 1; i < imu_data_segment.size(); ++i) {
                    Eigen::Matrix<double, 15, 15> Phi;
                    Eigen::Matrix<double, 15, 15> Cov =
                        state_->covariance.topLeftCorner<15, 15>();

                    const auto begin_imu =
                        std::dynamic_pointer_cast<DatasetIO::ImuData>(
                            imu_data_segment[i - 1]);
                    const auto end_imu =
                        std::dynamic_pointer_cast<DatasetIO::ImuData>(
                            imu_data_segment[i]);

                    assert(std::abs(begin_imu->timestamp - state_->timestamp) <
                           1e-6);

                    imu_propagator_->predict_and_compute(state_, begin_imu,
                                                         end_imu, &Phi, &Cov);

                    state_->covariance.topLeftCorner<15, 15>() = Cov;

                    state_->timestamp = end_imu->timestamp;

                    // propagate covariance of other states.
                    const int cov_size = state_->covariance.rows();
                    const int obs_size = cov_size - 15;
                    if (obs_size <= 0) {
                        continue;
                    }

                    state_->covariance.block(0, 15, 15, obs_size) =
                        Phi *
                        state_->covariance.block(0, 15, 15, obs_size).eval();
                    state_->covariance = state_->covariance.eval()
                                             .selfadjointView<Eigen::Upper>();
                }

                return true;
            }
        }
    } else {
        return false;
    }

    return false;
}

bool FilterFusionSystem::augment_state() {
    for (size_t d_i = 0; d_i < sync_data_.size(); d_i++) {
        if (sync_data_[d_i][0]->type == DatasetIO::MeasureType::kMonoImage) {
            CameraState::Ptr cam_state = std::make_shared<CameraState>();
            cam_state->timestamp = sync_data_[d_i][0]->timestamp;
            cam_state->id = cur_frame_id_;

            // Compute mean.
            cam_state->R_G_C =
                state_->imu_state_->R_G_I * state_->cam_extrinsics_[0]->R_B_S;
            cam_state->p_G_C =
                state_->imu_state_->p_G_I +
                state_->imu_state_->R_G_I * state_->cam_extrinsics_[0]->p_B_S;

            // Set index.
            cam_state->state_idx = state_->covariance.rows();

            // Push to state vector.
            state_->camera_states_.push_back(cam_state);

            // Extend covaraicne.
            const int old_size = state_->covariance.rows();
            const int new_size = old_size + 6;
            state_->covariance.conservativeResize(new_size, new_size);
            state_->covariance.block(old_size, 0, 6, new_size).setZero();
            state_->covariance.block(0, old_size, new_size, 6).setZero();

            /// Compute covariance.
            Eigen::Matrix<double, 6, 6> J_wrt_cam_pose;
            // J_wrt_wheel_pose << state->extrinsic.O_R_C.transpose(),
            // Eigen::Matrix3d::Zero(),
            //                     -state->wheel_pose.G_R_O *
            //                     TGK::Util::Skew(state->extrinsic.O_p_C),
            //                     Eigen::Matrix3d::Identity();

            const Eigen::Matrix<double, 6, 6> cov11 =
                state_->covariance.block<6, 6>(0, 0);
            state_->covariance.block<6, 6>(old_size, old_size) =
                J_wrt_cam_pose * cov11 * J_wrt_cam_pose.transpose();

            const auto& cov_top_rows =
                state_->covariance.block(0, 0, 6, old_size);
            // New lower line.
            state_->covariance.block(old_size, 0, 6, old_size) =
                J_wrt_cam_pose * cov_top_rows;

            // Force symmetric.
            state_->covariance =
                state_->covariance.eval().selfadjointView<Eigen::Lower>();
        }
    }

    cur_frame_id_++;

    return true;
}

bool FilterFusionSystem::update() {
    for (size_t d_i = 0; d_i < sync_data_.size(); d_i++) {
        if (sync_data_[d_i][0]->type == DatasetIO::MeasureType::kMonoImage) {
            //     // Track features.
            //     const auto image_type = img_ptr->type;
            //     std::vector<Eigen::Vector2d> tracked_pts;
            //     std::vector<long int> tracked_pt_ids;
            //     std::vector<long int> lost_pt_ids;
            //     std::set<long int> new_pt_ids;
            //     if (image_type == TGK::BaseType::MeasureType::kMonoImage) {
            //         feature_tracker_->TrackImage(img_ptr->image,
            //         &tracked_pts, &tracked_pt_ids, &lost_pt_ids,
            //         &new_pt_ids);
            //     } else if (image_type ==
            //     TGK::BaseType::MeasureType::kSimMonoImage) {
            //         const TGK::BaseType::SimMonoImageDataConstPtr sim_img_ptr
            //             = std::dynamic_pointer_cast<const
            //             TGK::BaseType::SimMonoImageData>(img_ptr);
            //         sim_feature_tracker_->TrackSimFrame(sim_img_ptr->features,
            //         sim_img_ptr->feature_ids,
            //                                             &tracked_pts,
            //                                             &tracked_pt_ids,
            //                                             &lost_pt_ids,
            //                                             &new_pt_ids);
            //     } else {
            //         LOG(ERROR) << "Not surpport image type.";
            //         exit(EXIT_FAILURE);
            //     }

            //     // Do not marginalize the last state if no enough camera
            //     state in the buffer. const bool marg_old_state =
            //     state_.camera_frames.size() >= config_.sliding_window_size_;

            //     // Update state.
            //     std::vector<Eigen::Vector2d> tracked_features;
            //     std::vector<Eigen::Vector2d> new_features;
            //     std::vector<Eigen::Vector3d> map_points;
            //     visual_updater_->UpdateState(img_ptr->image, marg_old_state,
            //                                  tracked_pts, tracked_pt_ids,
            //                                  lost_pt_ids, new_pt_ids,
            //                                  &state_, &tracked_features,
            //                                  &new_features, &map_points);
        }
    }

    return true;
}

bool FilterFusionSystem::marginalize_oldest_state() {
    // Marginalize old state.
    if (marg_old_state_) {
        if (state_->camera_states_.empty()) {
            return false;
        }

        // Remove camera state.
        int org_state_idx = state_->camera_states_.front()->state_idx;
        int state_idx = org_state_idx;
        state_->camera_states_.pop_front();
        for (auto& cam_state : state_->camera_states_) {
            cam_state->state_idx = state_idx;
            state_idx += cam_state->size;
        }

        // Remove row and col in covariance matrix.
        const int old_cov_size = state_->covariance.rows();
        const int new_cov_size = old_cov_size - 6;
        Eigen::MatrixXd new_cov(new_cov_size, new_cov_size);

        const Eigen::MatrixXd& old_cov = state_->covariance;
        new_cov.block<6, 6>(0, 0) = old_cov.block<6, 6>(0, 0);
        new_cov.block(0, 6, 6, new_cov_size - 6) =
            old_cov.block(0, 12, 6, new_cov_size - 6);
        new_cov.block(6, 6, new_cov_size - 6, new_cov_size - 6) =
            old_cov.block(12, 12, new_cov_size - 6, new_cov_size - 6);

        // Force symetric.
        state_->covariance = new_cov.selfadjointView<Eigen::Upper>();
    }

    return true;
}

}  // namespace SensorFusion
