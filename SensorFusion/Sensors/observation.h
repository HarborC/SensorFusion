/*
 * @Author: Jiagang Chen
 * @Date: 2021-11-04 06:56:42
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-04 07:07:12
 * @Description: ...
 * @Reference: ...
 */
#ifndef _SENSOR_FUSION_OBSERVATION_H_
#define _SENSOR_FUSION_OBSERVATION_H_

#include "../../common_header.h"

namespace SensorFusion {

struct Observation {
    typedef std::shared_ptr<Observation> Ptr;
    virtual ~Observation() {}

    DatasetIO::MeasureType type;
};

struct MonoImageObservation : public Observation {
    typedef std::shared_ptr<MonoImageObservation> Ptr;
    MonoImageObservation() { type = DatasetIO::MeasureType::kMonoImage; }
    ~MonoImageObservation() override = default;

    std::vector<Eigen::Vector2d>
};

struct StereoImageObservation : public Observation {
    typedef std::shared_ptr<StereoImageObservation> Ptr;
    StereoImageObservation() { type = DatasetIO::MeasureType::kStereoImage; }
    ~StereoImageObservation() override = default;
};

struct MultiImageObservation : public Observation {
    typedef std::shared_ptr<MultiImageObservation> Ptr;
    MultiImageObservation() { type = DatasetIO::MeasureType::kMultiImage; }
    ~MultiImageObservation() override = default;
};

}  // namespace SensorFusion

#endif