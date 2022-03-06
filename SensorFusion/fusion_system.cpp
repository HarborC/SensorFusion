#include "fusion_system.h"

namespace SensorFusion {

FusionSystem::FusionSystem(const std::string &confif_path) {
    system_config_ = std::make_shared<SystemConfig>(confif_path);
    data_base_ = std::make_shared<DataBase>();

    system_initializer_ = std::unique_ptr<SystemInitializer>(
        new SystemInitializer(system_config_, data_base_));
    filter_fusion_system_ = std::unique_ptr<FilterFusionSystem>(
        new FilterFusionSystem(system_config_, data_base_));
}

void FusionSystem::run() {
    while (1) {
        if (!system_initializer_->is_initialized()) {
            if (system_initializer_->initialize()) {
                system_initializer_->update_filter_state(
                    filter_fusion_system_->get_filter_state());
            }
        } else {
            if (filter_fusion_system_->get_measurement()) {
                if (!filter_fusion_system_->propagate())
                    continue;

                filter_fusion_system_->augment_state();

                filter_fusion_system_->update();

                filter_fusion_system_->marginalize_oldest_state();
            }
        }
    }
}

bool FusionSystem::feed_data(
    const std::vector<std::vector<DatasetIO::Measurement::Ptr>> &next_data) {
    for (size_t i = 0; i < next_data.size(); i++) {
        if (next_data[i][0]->type == DatasetIO::MeasureType::kMonoImage) {
            auto mono_data =
                std::dynamic_pointer_cast<DatasetIO::MonoImageData>(
                    next_data[i][0]);
            data_base_->feed_mono_image_data(mono_data);
        } else if (next_data[i][0]->type == DatasetIO::MeasureType::kAccel) {
            auto accel_data = std::dynamic_pointer_cast<DatasetIO::AccelData>(
                next_data[i][0]);
            auto gyro_data =
                std::dynamic_pointer_cast<DatasetIO::GyroData>(next_data[i][1]);
            data_base_->feed_imu_data(accel_data, gyro_data);
        } else if (next_data[i][0]->type == DatasetIO::MeasureType::kWheel) {
            auto wheel_data = std::dynamic_pointer_cast<DatasetIO::WheelData>(
                next_data[i][0]);
            data_base_->feed_wheel_data(wheel_data);
        } else if (next_data[i][0]->type == DatasetIO::MeasureType::kGnss) {
            auto gnss_data =
                std::dynamic_pointer_cast<DatasetIO::GnssData>(next_data[i][0]);
            data_base_->feed_gnss_data(gnss_data);
        }
    }

    // data_base_->print();
}

}  // namespace SensorFusion