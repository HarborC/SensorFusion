#ifndef _SENSOR_FUSION_IMU_H_
#define _SENSOR_FUSION_IMU_H_

#include "../../common_header.h"

using namespace DatasetIO;

namespace SensorFusion {

inline ImuData::Ptr interpolate_data(const ImuData::Ptr &imu_1,
                                     const ImuData::Ptr &imu_2,
                                     double timestamp);

std::vector<ImuData::Ptr> select_imu_readings(
    const std::deque<ImuData::Ptr> &imu_data, double time0, double time1,
    bool warn = true);

}  // namespace SensorFusion

#endif