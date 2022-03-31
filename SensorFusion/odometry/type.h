#pragma once

#include <iostream>
#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>

#include "utility.h"
#include "utils/lidar_utils.h"

namespace SensorFusion {

using KeypointId = size_t;
using CameraId = std::string;
using Feature = Eigen::Matrix<float, 7, 1>;
using Observation = std::pair<CameraId, Feature>;
using Correspond = std::pair<Eigen::Vector3d, Eigen::Vector3d>;
using Corresponds = std::vector<Correspond>;

struct ImageTrackerInput {
    using Ptr = std::shared_ptr<ImageTrackerInput>;

    double timestamp;
    std::unordered_map<std::string, cv::Mat> img_data;
};

struct ImageTrackerResult {
    using Ptr = std::shared_ptr<ImageTrackerResult>;

    double timestamp;
    std::map<KeypointId, std::vector<Observation>> observations;

    ImageTrackerInput::Ptr input_images;

    void show() {
        std::unordered_map<std::string, std::vector<cv::Point2f>> pts;
        if (input_images) {
            const auto& images = input_images->img_data;
            for (const auto& im : images) {
                pts[im.first] = std::vector<cv::Point2f>();
            }

            for (auto& obs : observations) {
                const auto& cam_id = obs.second[0].first;
                cv::Point2f pt;
                pt.x = (obs.second[0]).second(3);
                pt.y = (obs.second[0]).second(4);
                pts[cam_id].push_back(pt);
            }

            for (const auto& im : images) {
                const std::string& cam_id = im.first;
                if (!im.second.empty())
                    cv::imshow(cam_id + "_track_result",
                               utility::drawKeypoint1(im.second, pts[cam_id]));
            }
            cv::waitKey(10);
        }
    }
};

struct LidarFeatureResult {
    using Ptr = std::shared_ptr<LidarFeatureResult>;

    double timestamp;
    std::string sensor_id;
    PointCloudType::Ptr features;
    PointCloudType::Ptr filter_points;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
        filter_points_covariances;
};

}  // namespace SensorFusion