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

/** \brief imu data struct */
struct ImuData {
    using Ptr = std::shared_ptr<ImuData>;
    double timeStamp;
    Eigen::Vector3d accel, gyro;
    Eigen::Vector3d accel_bias, gyro_bias;
};

/** \brief point to line feature */
struct FeatureLine {
    Eigen::Vector3d pointOri;
    Eigen::Vector3d lineP1;
    Eigen::Vector3d lineP2;
    double error;
    bool valid;
    FeatureLine(Eigen::Vector3d po, Eigen::Vector3d p1, Eigen::Vector3d p2)
        : pointOri(std::move(po)),
          lineP1(std::move(p1)),
          lineP2(std::move(p2)) {
        valid = false;
        error = 0;
    }
    double ComputeError(const Eigen::Matrix4d& pose) {
        Eigen::Vector3d P_to_Map =
            pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
        double l12 =
            std::sqrt((lineP1(0) - lineP2(0)) * (lineP1(0) - lineP2(0)) +
                      (lineP1(1) - lineP2(1)) * (lineP1(1) - lineP2(1)) +
                      (lineP1(2) - lineP2(2)) * (lineP1(2) - lineP2(2)));
        double a012 = std::sqrt(
            ((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) -
             (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1))) *
                ((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) -
                 (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1))) +
            ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) -
             (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2))) *
                ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) -
                 (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2))) +
            ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) -
             (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2))) *
                ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) -
                 (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2))));
        error = a012 / l12;
    }
};

/** \brief point to plan feature */
struct FeaturePlan {
    Eigen::Vector3d pointOri;
    double pa;
    double pb;
    double pc;
    double pd;
    double error;
    bool valid;
    FeaturePlan(const Eigen::Vector3d& po, const double& pa_, const double& pb_,
                const double& pc_, const double& pd_)
        : pointOri(po), pa(pa_), pb(pb_), pc(pc_), pd(pd_) {
        valid = false;
        error = 0;
    }
    double ComputeError(const Eigen::Matrix4d& pose) {
        Eigen::Vector3d P_to_Map =
            pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
        error = pa * P_to_Map(0) + pb * P_to_Map(1) + pc * P_to_Map(2) + pd;
    }
};

/** \brief point to plan feature */
struct FeaturePlanVec {
    Eigen::Vector3d pointOri;
    Eigen::Vector3d pointProj;
    Eigen::Matrix3d sqrt_info;
    double error;
    bool valid;
    FeaturePlanVec(const Eigen::Vector3d& po, const Eigen::Vector3d& p_proj,
                   Eigen::Matrix3d sqrt_info_)
        : pointOri(po), pointProj(p_proj), sqrt_info(sqrt_info_) {
        valid = false;
        error = 0;
    }
    double ComputeError(const Eigen::Matrix4d& pose) {
        Eigen::Vector3d P_to_Map =
            pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
        error = (P_to_Map - pointProj).norm();
    }
};

/** \brief non feature */
struct FeatureNon {
    Eigen::Vector3d pointOri;
    double pa;
    double pb;
    double pc;
    double pd;
    double error;
    bool valid;
    FeatureNon(const Eigen::Vector3d& po, const double& pa_, const double& pb_,
               const double& pc_, const double& pd_)
        : pointOri(po), pa(pa_), pb(pb_), pc(pc_), pd(pd_) {
        valid = false;
        error = 0;
    }
    double ComputeError(const Eigen::Matrix4d& pose) {
        Eigen::Vector3d P_to_Map =
            pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
        error = pa * P_to_Map(0) + pb * P_to_Map(1) + pc * P_to_Map(2) + pd;
    }
};

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
};

}  // namespace SensorFusion