#pragma once

#include "../../image_tracker.h"
#include "../frame.h"
#include "../sensor_flag.h"

namespace SensorFusion {

/** \brief camera frame structure */
struct CameraFrame : public Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<CameraFrame>;
    CameraFrame() : Frame() { sensorType = SensorFlag::CAMERA; }
    ~CameraFrame() override = default;

    ImageTrackerResult::Ptr trackResult;
};

}  // namespace SensorFusion