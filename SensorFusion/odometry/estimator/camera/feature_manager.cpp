/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace SensorFusion {

float FOCAL_LENGTH = 500;
float MIN_PARALLAX = 10;
int MIN_TRACK_FRAME_NUM = 4;

std::unordered_map<std::string, std::vector<cv::Point2f>>
FeatureManager::showFrame(int frame_idx) {
    std::unordered_map<std::string, std::vector<cv::Point2f>> pts;
    for (auto &it : feature) {
        if (it.start_frame <= frame_idx && it.endFrame() >= frame_idx) {
            int idx = frame_idx - it.start_frame;
            const auto &cam_id = it.feature_per_frame[idx].camids[0];
            if (pts.find(cam_id) != pts.end()) {
                cv::Point2f pt;
                pt.x = it.feature_per_frame[idx].uvs[0].x();
                pt.y = it.feature_per_frame[idx].uvs[0].y();
                pts[cam_id].push_back(pt);
            } else {
                std::vector<cv::Point2f> pp;
                cv::Point2f pt;
                pt.x = it.feature_per_frame[idx].uvs[0].x();
                pt.y = it.feature_per_frame[idx].uvs[0].y();
                pp.push_back(pt);
                pts[cam_id] = pp;
            }
        }
    }
    return pts;
}

bool FeatureManager::addFeature(
    int frame_count,
    const std::map<KeypointId, std::vector<Observation>> &image, double td) {
    double parallax_sum = 0;
    int parallax_num = 0;

    last_track_num = 0;
    last_average_parallax = 0;
    new_feature_num = 0;
    long_track_num = 0;

    for (auto &id_pts : image) {
        const double td0 = 0.0;
        const auto &cam_id0 = id_pts.second[0].first;
        const auto &feature0 = id_pts.second[0].second;
        FeaturePerFrame f_per_fra(cam_id0, feature0, td0);
        if (id_pts.second.size() > 1)
            for (size_t i = 1; i < id_pts.second.size(); i++) {
                const double td1 = 0.0;
                const auto &cam_id = id_pts.second[i].first;
                const auto &feature = id_pts.second[i].second;
                f_per_fra.addStereoObservation(cam_id, feature, td1);
            }

        const auto &feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(),
                          [feature_id](const FeaturePerId &it) {
                              return it.feature_id == feature_id;
                          });

        if (it == feature.end()) {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
            new_feature_num++;
        } else if (it->feature_id == feature_id) {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            if (it->feature_per_frame.size() >= MIN_TRACK_FRAME_NUM)
                long_track_num++;
        }
    }

    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 ||
        new_feature_num > 0.5 * last_track_num)
        return true;

    for (auto &it_per_id : feature) {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) -
                    1 >=
                frame_count - 1) {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0) {
        return true;
    } else {
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
        return parallax_sum / parallax_num >= MIN_PARALLAX;

        // return true;
    }
}

void FeatureManager::removeOutlier(std::set<int> &outlierIndex) {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) {
        it_next++;
        int index = it->feature_id;
        if (outlierIndex.find(index) != outlierIndex.end()) {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBack() {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count) {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) {
        it_next++;

        if (it->start_frame == frame_count) {
            it->start_frame--;
        } else {
            // int j = WINDOW_SIZE - 1 - it->start_frame;
            int j = frame_count - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFailures() {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

Corresponds FeatureManager::getCorresponds(const std::string &cam_id,
                                           int frame_count_l,
                                           int frame_count_r) {
    Corresponds corres;
    for (auto &it : feature) {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r &&
            it.feature_per_frame[0].camids[0] == cam_id) {
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;
            corres.push_back(
                std::make_pair(it.feature_per_frame[idx_l].points[0],
                               it.feature_per_frame[idx_r].points[0]));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const Eigen::VectorXd &x) {
    int feature_index = -1;
    for (auto &it_per_id : feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < MIN_TRACK_FRAME_NUM)
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        if (it_per_id.estimated_depth < 0)
            it_per_id.solve_flag = 2;
        else
            it_per_id.solve_flag = 1;
    }
}

Eigen::VectorXd FeatureManager::getDepthVector() {
    Eigen::VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < MIN_TRACK_FRAME_NUM)
            continue;
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
    }
    return dep_vec;
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R,
                                          Eigen::Vector3d marg_P,
                                          Eigen::Matrix3d new_R,
                                          Eigen::Vector3d new_P) {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].points[0];
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2) {
                feature.erase(it);
                continue;
            } else {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j.norm();
                it->estimated_depth = dep_j;
            }
        }
    }
}

void FeatureManager::triangulatePoint(
    const std::vector<Eigen::Matrix<double, 3, 4>> &poses,
    const std::vector<Eigen::Vector3d> &points2d, Eigen::Vector3d &point3d) {
    Eigen::MatrixXd design_matrix(poses.size() * 2, 4);
    for (size_t i = 0; i < poses.size(); i++) {
        design_matrix.row(i * 2 + 0) =
            points2d[i](0) * poses[i].row(2) - points2d[i](2) * poses[i].row(0);
        design_matrix.row(i * 2 + 1) =
            points2d[i](1) * poses[i].row(2) - points2d[i](2) * poses[i].row(1);
    }

    Eigen::Vector4d triangulated_point =
        design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point3d(0) = triangulated_point(0) / triangulated_point(3);
    point3d(1) = triangulated_point(1) / triangulated_point(3);
    point3d(2) = triangulated_point(2) / triangulated_point(3);
}

void FeatureManager::triangulate(Eigen::Vector3d twi[], Eigen::Matrix3d Rwi[]) {
    const size_t max_triangulate_size = 3;
    for (auto &it_per_id : feature) {
        if (it_per_id.estimated_depth != std::numeric_limits<double>::max())
            continue;

        if (it_per_id.feature_per_frame[0].is_stereo) {
            int idx0 = it_per_id.start_frame;
            std::vector<Eigen::Matrix<double, 3, 4>> poses;
            std::vector<Eigen::Vector3d> points2d;
            for (size_t i = 0; i < max_triangulate_size &&
                               i < it_per_id.feature_per_frame[0].camids.size();
                 i++) {
                auto cam_id = it_per_id.feature_per_frame[0].camids[i];
                Eigen::Matrix<double, 3, 4> pose =
                    Eigen::Matrix<double, 3, 4>::Zero();
                Eigen::Matrix4d Tic = calib->Tic[cam_id];
                Eigen::Vector3d t =
                    twi[idx0] + Rwi[idx0] * Tic.block<3, 1>(0, 3);
                Eigen::Matrix3d R = Rwi[idx0] * Tic.block<3, 3>(0, 0);
                pose.leftCols<3>() = R.transpose();
                pose.rightCols<1>() = -R.transpose() * t;
                poses.push_back(pose);

                const auto &pt = it_per_id.feature_per_frame[0].points[i];
                points2d.push_back(pt);
            }

            Eigen::Vector3d point3d;
            triangulatePoint(poses, points2d, point3d);

            Eigen::Vector3d localPoint =
                poses[0].leftCols<3>() * point3d + poses[0].rightCols<1>();
            it_per_id.estimated_depth = localPoint.norm();

            continue;

        } else if (it_per_id.feature_per_frame.size() > 1) {
            int idx0 = it_per_id.start_frame;
            std::vector<Eigen::Matrix<double, 3, 4>> poses;
            std::vector<Eigen::Vector3d> points2d;
            for (size_t i = 0; i < max_triangulate_size &&
                               i < it_per_id.feature_per_frame.size();
                 i++) {
                auto cam_id = it_per_id.feature_per_frame[i].camids[0];
                Eigen::Matrix<double, 3, 4> pose =
                    Eigen::Matrix<double, 3, 4>::Zero();
                Eigen::Matrix4d Tic = calib->Tic[cam_id];
                Eigen::Vector3d t =
                    twi[idx0] + Rwi[idx0] * Tic.block<3, 1>(0, 3);
                Eigen::Matrix3d R = Rwi[idx0] * Tic.block<3, 3>(0, 0);
                pose.leftCols<3>() = R.transpose();
                pose.rightCols<1>() = -R.transpose() * t;
                poses.push_back(pose);

                const auto &pt = it_per_id.feature_per_frame[i].points[0];
                points2d.push_back(pt);

                idx0++;
            }

            Eigen::Vector3d point3d;
            triangulatePoint(poses, points2d, point3d);

            Eigen::Vector3d localPoint =
                poses[0].leftCols<3>() * point3d + poses[0].rightCols<1>();
            it_per_id.estimated_depth = localPoint.norm();

            continue;
        }

        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < MIN_TRACK_FRAME_NUM)
            continue;

        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);

        int idx0 = it_per_id.start_frame, idx1 = idx0 - 1;
        auto cam_id0 = it_per_id.feature_per_frame[0].camids[0];
        auto Tic0 = calib->Tic[cam_id0];

        Eigen::Vector3d t0 = twi[idx0] + Rwi[idx0] * Tic0.block<3, 1>(0, 3);
        Eigen::Matrix3d R0 = Rwi[idx0] * Tic0.block<3, 3>(0, 0);

        int svd_idx = 0;
        for (size_t i = 0; i < it_per_id.feature_per_frame.size(); i++) {
            auto &it_per_frame = it_per_id.feature_per_frame[i];

            idx1++;

            auto cam_id1 = it_per_frame.camids[0];
            auto Tic1 = calib->Tic[cam_id1];

            Eigen::Vector3d t1 = twi[idx1] + Rwi[idx1] * Tic1.block<3, 1>(0, 3);
            Eigen::Matrix3d R1 = Rwi[idx1] * Tic1.block<3, 3>(0, 0);
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.points[0].normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (idx0 == idx1)
                continue;
        }

        Eigen::Vector4d svd_V =
            Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV)
                .matrixV()
                .rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        it_per_id.estimated_depth = svd_method;

        if (it_per_id.estimated_depth < 0.1) {
            it_per_id.estimated_depth = std::numeric_limits<double>::max();
        }
    }
}

bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P,
                                    std::vector<cv::Point2f> &pts2D,
                                    std::vector<cv::Point3f> &pts3D) {
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    // printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4) {
        printf(
            "feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    // pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 /
    // focalLength, 0.99, inliers);

    if (!pnp_succ) {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r);
    // cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

void FeatureManager::initFramePoseByPnP(int frameCnt, Eigen::Vector3d twi[],
                                        Eigen::Matrix3d Rwi[]) {
    if (frameCnt > 0) {
        std::unordered_map<CameraId, std::pair<std::vector<cv::Point3f>,
                                               std::vector<cv::Point2f>>>
            corres;
        for (auto &it_per_id : feature) {
            auto cam_id = it_per_id.feature_per_frame[0].camids[0];
            auto &Tic = calib->Tic[cam_id];
            if (it_per_id.estimated_depth !=
                std::numeric_limits<double>::max()) {
                int index = frameCnt - it_per_id.start_frame;
                if ((int)it_per_id.feature_per_frame.size() >= index + 1) {
                    Eigen::Vector3d ptsInCam =
                        Tic.block<3, 3>(0, 0) *
                            (it_per_id.feature_per_frame[0].points[0] *
                             it_per_id.estimated_depth) +
                        Tic.block<3, 1>(0, 3);
                    Eigen::Vector3d ptsInWorld =
                        Rwi[it_per_id.start_frame] * ptsInCam +
                        twi[it_per_id.start_frame];

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(),
                                        ptsInWorld.z());
                    cv::Point2f point2d(
                        it_per_id.feature_per_frame[index].points[0].x() /
                            it_per_id.feature_per_frame[index].points[0].z(),
                        it_per_id.feature_per_frame[index].points[0].y() /
                            it_per_id.feature_per_frame[index].points[0].z());
                    corres[cam_id].first.push_back(point3d);
                    corres[cam_id].second.push_back(point2d);
                }
            }
        }

        for (auto &co : corres) {
            auto &cam_id = co.first;
            auto &pts2D = co.second.second;
            auto &pts3D = co.second.first;
            auto &Tic = calib->Tic[cam_id];

            Eigen::Matrix3d RCam;
            Eigen::Vector3d PCam;
            // trans to w_T_cam
            RCam = Rwi[frameCnt - 1] * Tic.block<3, 3>(0, 0);
            PCam =
                Rwi[frameCnt - 1] * Tic.block<3, 1>(0, 3) + twi[frameCnt - 1];
            if (solvePoseByPnP(RCam, PCam, pts2D, pts3D)) {
                // trans to w_T_imu
                Rwi[frameCnt] = RCam * Tic.block<3, 3>(0, 0).transpose();
                twi[frameCnt] = -RCam * Tic.block<3, 3>(0, 0).transpose() *
                                    Tic.block<3, 1>(0, 3) +
                                PCam;

                break;
            }
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id,
                                            int frame_count) {
    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i =
        it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j =
        it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.points[0];

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.points[0];
    Vector3d p_i_comp;

    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv,
                            du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}

}  // namespace SensorFusion