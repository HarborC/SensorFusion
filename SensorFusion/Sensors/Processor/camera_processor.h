/*
 * @Author: Jiagang Chen
 * @Date: 2021-11-04 05:23:52
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-04 11:34:19
 * @Description: ...
 * @Reference: ...
 */
#ifndef _SENSOR_FUSION_CAMERA_PROCESSOR_H_
#define _SENSOR_FUSION_CAMERA_PROCESSOR_H_

#include "../../Config/config.h"
#include "../../Data/database.h"
#include "../../common_header.h"
#include "../observation.h"

#include "../Camera/Tracker/TrackBase.h"

namespace SensorFusion {

class CameraProcessor {
public:
    std::shared_ptr<CameraProcessor> Ptr;
    CameraProcessor(const SystemConfig::Ptr& system_config,
                    const DataBase::Ptr& data_base);
    ~CameraProcessor() {}

    bool track_feature(const Measurement::Ptr& measurement,
                       const std::vector<cv::Mat>& masks = {});

protected:
    cv::Mat create_mask(const size_t& cam_id);

protected:
    SystemConfig::Ptr system_config_;
    DataBase::Ptr data_base_;

    std::shared_ptr<ov_core::FeatureDatabase> track_database_;
    std::shared_ptr<ov_core::TrackBase> track_feats_;
    std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>>
        track_cameras_;
    int max_num_pts_ = 250;
    bool use_stereo_ = true;
    bool downsample_cameras_ = true;
    ov_core::TrackBase::HistogramMethod method_hist_ =
        ov_core::TrackBase::HistogramMethod::CLAHE;
    int fast_threshold_ = 20;
    int min_px_dist_ = 10;
    int grid_x_ = 5;
    int grid_y_ = 5;
};

}  // namespace SensorFusion

#endif