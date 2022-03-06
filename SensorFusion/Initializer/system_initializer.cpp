#include "system_initializer.h"

namespace SensorFusion {

SystemInitializer::SystemInitializer(const SystemConfig::Ptr& system_config,
                                     const DataBase::Ptr& data_base)
    : system_config_(system_config),
      data_base_(data_base),
      initialized_(false) {
    Eigen::Matrix4d T_cam0_imu =
        system_config_->state_config_.T_imu0_cam0.inverse();

    vio_flexible_initializer_ = std::make_shared<larvio::FlexibleInitializer>(
        system_config_->state_config_.zupt_max_feature_dis,
        system_config_->state_config_.static_num,
        system_config_->state_config_.td_cam0, system_config_->state_config_.Ma,
        system_config_->state_config_.Tg, system_config_->state_config_.As,
        sqrt(system_config_->state_config_.imu_acc_noise),
        sqrt(system_config_->state_config_.imu_acc_bias_noise),
        sqrt(system_config_->state_config_.imu_gyro_noise),
        sqrt(system_config_->state_config_.imu_gyro_bias_noise),
        T_cam0_imu.block<3, 3>(0, 0), T_cam0_imu.block<3, 1>(0, 3),
        1 / (2 * system_config_->state_config_.imu_rate));
}

bool SystemInitializer::is_initialized() { return initialized_; }

bool SystemInitializer::get_measurement() {
    if (system_config_->system_initializer_config_.initializer_type_ == 1) {
        if (data_base_->get_sync_data(sync_data_)) {
            return true;
        }
    }

    return false;
}

bool SystemInitializer::initialize() {
    if (system_config_->system_initializer_config_.initializer_type_ == 1) {
        if (get_measurement()) {
            std::vector<larvio::ImuData> imu_msg_buffer;

            for (size_t d_i = 0; d_i < sync_data_.size(); d_i++) {
                if (sync_data_[d_i][0]->type == DatasetIO::MeasureType::kImu) {
                    auto& imu_data_segment = sync_data_[d_i];

                    // propagate state.
                    for (size_t i = 1; i < imu_data_segment.size(); ++i) {
                        auto imu_temp_ptr =
                            std::dynamic_pointer_cast<DatasetIO::ImuData>(
                                imu_data_segment[i]);

                        larvio::ImuData imu_temp(
                            imu_temp_ptr->timestamp, imu_temp_ptr->wm(0),
                            imu_temp_ptr->wm(1), imu_temp_ptr->wm(2),
                            imu_temp_ptr->am(0), imu_temp_ptr->am(1),
                            imu_temp_ptr->am(2));
                        imu_msg_buffer.push_back(imu_temp);
                    }
                }
            }

            larvio::MonoCameraMeasurementPtr mono_image_msg;

            if (vio_flexible_initializer_->tryIncInit(
                    imu_msg_buffer, mono_image_msg, m_gyro_old, m_acc_old,
                    vio_initializer_state_)) {
                initialized_ = true;

                // // Set take off time
                // take_off_stamp = state_server.imu_state.time;
                // // Set last time of last ZUPT
                // last_ZUPT_time = state_server.imu_state.time;
                // // Initialize time of last update
                // last_update_time = state_server.imu_state.time;
            } else
                return false;
        }
    } else if (system_config_->system_initializer_config_.initializer_type_ ==
               0) {
        return false;
    }

    return true;
}

bool SystemInitializer::update_filter_state(FilterState::Ptr state) {
    if (initialized_) {
        if (system_config_->system_initializer_config_.initializer_type_ == 1) {
            state->timestamp = vio_initializer_state_.time;
            state->imu_state_->bg = vio_initializer_state_.gyro_bias;
            state->imu_state_->ba = vio_initializer_state_.acc_bias;
            state->imu_state_->R_G_I = (larvio::quaternionToRotation(
                                            vio_initializer_state_.orientation))
                                           .transpose();
            state->imu_state_->p_G_I = vio_initializer_state_.position;
            state->imu_state_->v_G_I = vio_initializer_state_.velocity;

            return true;
        }
    }

    return false;
}

}  // namespace SensorFusion