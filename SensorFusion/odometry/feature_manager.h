/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <limits.h>
#include <algorithm>
#include <list>
#include <numeric>
#include <set>
#include <vector>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#include "calibration.hpp"
#include "type.h"
#include "utility.h"

namespace SensorFusion {

class FeaturePerFrame {
public:
    FeaturePerFrame(const CameraId &cam_id, const Feature &feature, double td) {
        camids.push_back(cam_id);

        Eigen::Vector3d point;
        point.x() = feature(0);
        point.y() = feature(1);
        point.z() = feature(2);
        points.push_back(point);

        Eigen::Vector2d uv;
        uv.x() = feature(3);
        uv.y() = feature(4);
        uvs.push_back(uv);

        double cur_td = td;
        cur_tds.push_back(cur_td);

        is_stereo = false;
    }

    void addStereoObservation(const CameraId &cam_id, const Feature &feature,
                              double td) {
        camids.push_back(cam_id);

        Eigen::Vector3d point;
        point.x() = feature(0);
        point.y() = feature(1);
        point.z() = feature(2);
        points.push_back(point);

        Eigen::Vector2d uv;
        uv.x() = feature(3);
        uv.y() = feature(4);
        uvs.push_back(uv);

        double cur_td = td;
        cur_tds.push_back(cur_td);

        is_stereo = true;
    }

    std::vector<double> cur_tds;
    std::vector<CameraId> camids;
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector2d> uvs;
    bool is_stereo;
};

class FeaturePerId {
public:
    const int feature_id;
    int start_frame;
    int used_num;
    double estimated_depth;
    int solve_flag;  // 0 haven't solve yet; 1 solve succ; 2 solve fail;
    std::vector<FeaturePerFrame> feature_per_frame;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id),
          start_frame(_start_frame),
          used_num(0),
          estimated_depth(std::numeric_limits<double>::max()),
          solve_flag(0) {}

    int endFrame() { return start_frame + feature_per_frame.size() - 1; }
};

class FeatureManager {
public:
    FeatureManager() {}

    // Deal With Feature
    bool addFeature(int frame_count,
                    const std::map<KeypointId, std::vector<Observation>> &image,
                    double td);
    Corresponds getCorresponds(const std::string &cam_id, int frame_count_l,
                               int frame_count_r);
    void clearFeature() { feature.clear(); };
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier(std::set<int> &outlierIndex);
    int getFeatureCount() {
        int cnt = 0;
        for (auto &it : feature) {
            it.used_num = it.feature_per_frame.size();
            if (it.used_num >= 4)
                cnt++;
        }
        return cnt;
    }

    // Deal With Depth
    void removeFailures();
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P,
                              Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void setDepth(const Eigen::VectorXd &x);
    Eigen::VectorXd getDepthVector();
    void clearDepth() {
        for (auto &it_per_id : feature)
            it_per_id.estimated_depth = std::numeric_limits<double>::max();
    }

    // Triangulate
    void triangulate(Eigen::Vector3d twi[], Eigen::Matrix3d Rwi[]);
    void triangulatePoint(const std::vector<Eigen::Matrix<double, 3, 4>> &poses,
                          const std::vector<Eigen::Vector3d> &points2d,
                          Eigen::Vector3d &point3d);

    // PnP
    void initFramePoseByPnP(int frameCnt, Eigen::Vector3d Ps[],
                            Eigen::Matrix3d Rs[]);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                        std::vector<cv::Point2f> &pts2D,
                        std::vector<cv::Point3f> &pts3D);

    // Viz
    std::unordered_map<std::string, std::vector<cv::Point2f>> showFrame(
        int frame_idx);

public:
    std::list<FeaturePerId> feature;
    int last_track_num;
    int new_feature_num;
    int long_track_num;
    double last_average_parallax;

    Calibration::Ptr calib;

private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
};

}  // namespace SensorFusion