/*
 * @Author: Jiagang Chen
 * @Date: 2021-12-09 17:41:59
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-12-09 17:44:32
 * @Description: ...
 * @Reference: ...
 */
#ifndef _VISUALIZER_POSE_H_
#define _VISUALIZER_POSE_H_

#include "Common.h"

namespace Visualizer {

struct Pose
{
    double timestamp_;
    Eigen::Matrix3d R_;
    Eigen::Vector3d t_;

    Pose() : timestamp_(-1), R_(Eigen::Matrix3d::Identity()), t_(Eigen::Vector3d::Zero()) {}
    Pose(const double& timestamp, const Eigen::Matrix3d &R, const Eigen::Vector3d &t) : timestamp_(timestamp), R_(R), t_(t) {}
    Pose(const double& timestamp, const Eigen::Matrix4d &T) : timestamp_(timestamp) {
        R_ = T.block<3,3>(0,0);
        t_ = T.block<3,1>(0,3);
    };
};

}

#endif