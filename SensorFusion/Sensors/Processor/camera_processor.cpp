/*
 * @Author: Jiagang Chen
 * @Date: 2021-11-04 05:27:28
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-04 09:19:17
 * @Description: ...
 * @Reference: ...
 */
#include "camera_processor.h"

#include "../Camera/Tracker/TrackKLT.h"

namespace SensorFusion {

CameraProcessor::CameraProcessor(const SystemConfig::Ptr& system_config,
                                 const DataBase::Ptr& data_base)
    : system_config_(system_config), data_base_(data_base) {
    for (size_t i = 0; i < system_config_->sensor_config_.camera_models_.size();
         i++) {
        track_cameras_[i] = std::make_shared<ov_core::CamBase>(
            system_config_->sensor_config_.camera_models_[i]);
    }

    track_database_ = std::make_shared<ov_core::FeatureDatabase>();
    track_feats_ = std::shared_ptr<ov_core::TrackBase>(new ov_core::TrackKLT(
        track_cameras_, max_num_pts_, 0, use_stereo_, method_hist_,
        fast_threshold_, grid_x_, grid_y_, min_px_dist_));
}

bool CameraProcessor::track_feature(const Measurement::Ptr& measurement,
                                    const std::vector<cv::Mat>& masks) {
    ov_core::CameraData message;

    if (measurement->type == DatasetIO::MeasureType::kMonoImage) {
        auto mono_image_measurement =
            std::dynamic_pointer_cast<DatasetIO::MonoImageData>(measurement);
        message.sensor_ids.resize(1);
        message.images.resize(1);
        message.masks.resize(1);
        message.sensor_ids[0] = 0;
        message.timestamp = mono_image_measurement->timestamp;

        // Downsample if we are downsampling
        cv::Mat img = mono_image_measurement->data.clone();
        cv::Mat mask =
            masks.size() >= 1 && !masks[0].empty() ? masks[0] : create_mask(0);
        if (downsample_cameras_) {
            cv::pyrDown(img, img, cv::Size(img.cols / 2.0, img.rows / 2.0));
            cv::pyrDown(mask, mask, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
        }
        message.images[0] = img;
        message.masks[0] = mask;
    } else if (measurement->type == DatasetIO::MeasureType::kStereoImage) {
        auto stereo_image_measurement =
            std::dynamic_pointer_cast<DatasetIO::StereoImageData>(measurement);
        message.sensor_ids.resize(2);
        message.images.resize(2);
        message.masks.resize(2);
        message.sensor_ids[0] = 0;
        message.sensor_ids[1] = 1;
        message.timestamp = stereo_image_measurement->timestamp;

        for (int i = 0; i < 2; i++) {
            // Downsample if we are downsampling
            cv::Mat img = stereo_image_measurement->data[i].clone();
            cv::Mat mask = masks.size() >= i && !masks[i].empty()
                               ? masks[i]
                               : create_mask(i);
            if (downsample_cameras_) {
                cv::pyrDown(img, img, cv::Size(img.cols / 2.0, img.rows / 2.0));
                cv::pyrDown(mask, mask,
                            cv::Size(mask.cols / 2.0, mask.rows / 2.0));
            }
            message.images[i] = img;
            message.masks[i] = mask;
        }

    } else if (measurement->type == DatasetIO::MeasureType::kMultiImage) {
        auto multi_image_measurement =
            std::dynamic_pointer_cast<DatasetIO::MultiImageData>(measurement);
        message.sensor_ids.resize(multi_image_measurement->data.size());
        message.images.resize(multi_image_measurement->data.size());
        message.masks.resize(multi_image_measurement->data.size());
        for (size_t i = 0; i < multi_image_measurement->data.size(); i++) {
            message.sensor_ids[i] = i;
        }
        message.timestamp = multi_image_measurement->timestamp;

        for (int i = 0; i < multi_image_measurement->data.size(); i++) {
            // Downsample if we are downsampling
            cv::Mat img = multi_image_measurement->data[i].clone();
            cv::Mat mask = masks.size() >= i && !masks[i].empty()
                               ? masks[i]
                               : create_mask(i);
            if (downsample_cameras_) {
                cv::pyrDown(img, img, cv::Size(img.cols / 2.0, img.rows / 2.0));
                cv::pyrDown(mask, mask,
                            cv::Size(mask.cols / 2.0, mask.rows / 2.0));
            }
            message.images[i] = img;
            message.masks[i] = mask;
        }
    } else {
        return false;
    }

    // Perform our feature tracking!
    track_feats_->feed_new_camera(message);
    track_database_->append_new_measurements(
        track_feats_->get_feature_database());

    return true;
}

cv::Mat CameraProcessor::create_mask(const size_t& cam_id) {
    return cv::Mat(track_cameras_[cam_id]->camera_ptr->imageHeight(),
                   track_cameras_[cam_id]->camera_ptr->imageWidth(), CV_8UC1,
                   cv::Scalar(255));
}

}  // namespace SensorFusion
