#ifndef _SENSOR_FUSION_FUSION_SYSTEM_H_
#define _SENSOR_FUSION_FUSION_SYSTEM_H_

#include "Data/database.h"
#include "Filter/filter_fusion_system.h"
#include "Initializer/system_initializer.h"
#include "common_header.h"

namespace SensorFusion {

class FusionSystem {
public:
    FusionSystem(const std::string &confif_path);
    ~FusionSystem() {}
    void run();
    bool feed_data(
        const std::vector<std::vector<DatasetIO::Measurement::Ptr>> &next_data);

public:
    std::unique_ptr<SystemInitializer> system_initializer_;
    std::unique_ptr<FilterFusionSystem> filter_fusion_system_;
    std::shared_ptr<SystemConfig> system_config_;
    std::shared_ptr<DataBase> data_base_;
};

}  // namespace SensorFusion

#endif