#ifndef _SENSOR_FUSION_SYSTEM_INITIALIZER_H_
#define _SENSOR_FUSION_SYSTEM_INITIALIZER_H_

#include "../Config/config.h"
#include "../Data/database.h"
#include "../common_header.h"

#include "../Filter/State/filter_state.h"
#include "Vio/flexible_initializer.h"

namespace SensorFusion {

class SystemInitializer {
public:
    std::shared_ptr<SystemInitializer> Ptr;
    SystemInitializer(const SystemConfig::Ptr& system_config,
                      const DataBase::Ptr& data_base);
    ~SystemInitializer() {}
    bool is_initialized();
    bool get_measurement();
    bool initialize();

    bool update_filter_state(FilterState::Ptr state);

protected:
    bool initialized_;
    SystemConfig::Ptr system_config_;
    DataBase::Ptr data_base_;

protected:
    std::vector<std::vector<DatasetIO::Measurement::Ptr>> sync_data_;

protected:
    std::shared_ptr<larvio::FlexibleInitializer> vio_flexible_initializer_;
    larvio::IMUState vio_initializer_state_;
    Eigen::Vector3d m_gyro_old, m_acc_old;
};

}  // namespace SensorFusion

#endif