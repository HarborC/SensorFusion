/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "initial_sfm.h"

GlobalSFM::GlobalSFM() {}

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                                 Eigen::Matrix<double, 3, 4> &Pose1,
                                 Eigen::Vector2d &point0,
                                 Eigen::Vector2d &point1,
                                 Eigen::Vector3d &point_3d) {
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point =
        design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool GlobalSFM::solveFrameByPnP(Eigen::Matrix3d &R_initial,
                                Eigen::Vector3d &P_initial, int i,
                                std::vector<SFMFeature> &sfm_f) {
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;
    for (int j = 0; j < feature_num; j++) {
        if (sfm_f[j].state != true)
            continue;
        Eigen::Vector2d point2d;
        for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) {
            if (sfm_f[j].observation[k].first == i) {
                Eigen::Vector2d img_pts = sfm_f[j].observation[k].second;
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
                cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1],
                                  sfm_f[j].position[2]);
                pts_3_vector.push_back(pts_3);
                break;
            }
        }
    }
    if (int(pts_2_vector.size()) < 15) {
        printf("unstable features tracking, please slowly move you device!\n");
        if (int(pts_2_vector.size()) < 10)
            return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
    if (!pnp_succ) {
        return false;
    }
    cv::Rodrigues(rvec, r);
    // cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;
}

void GlobalSFM::triangulateTwoFrames(int frame0,
                                     Eigen::Matrix<double, 3, 4> &Pose0,
                                     int frame1,
                                     Eigen::Matrix<double, 3, 4> &Pose1,
                                     std::vector<SFMFeature> &sfm_f) {
    assert(frame0 != frame1);
    for (int j = 0; j < feature_num; j++) {
        if (sfm_f[j].state == true)
            continue;
        bool has_0 = false, has_1 = false;
        Eigen::Vector2d point0;
        Eigen::Vector2d point1;
        for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) {
            if (sfm_f[j].observation[k].first == frame0) {
                point0 = sfm_f[j].observation[k].second;
                has_0 = true;
            }
            if (sfm_f[j].observation[k].first == frame1) {
                point1 = sfm_f[j].observation[k].second;
                has_1 = true;
            }
        }
        if (has_0 && has_1) {
            Eigen::Vector3d point_3d;
            triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
            // cout << "trangulated : " << frame1 << "  3d point : "  << j << "
            // " << point_3d.transpose() << endl;
        }
    }
}

// q w_R_cam t w_R_cam
// c_rotation cam_R_w
// c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(std::vector<Eigen::Quaterniond> &q,
                          std::vector<Eigen::Vector3d> &T, int l,
                          const Eigen::Matrix3d relative_R,
                          const Eigen::Vector3d relative_T,
                          std::vector<SFMFeature> &sfm_f,
                          std::map<int, Eigen::Vector3d> &sfm_tracked_points) {
    feature_num = sfm_f.size();
    const int frame_num = q.size();
    // intial two view
    q[l].w() = 1;
    q[l].x() = 0;
    q[l].y() = 0;
    q[l].z() = 0;
    T[l].setZero();
    q[frame_num - 1] = q[l] * Eigen::Quaterniond(relative_R);
    T[frame_num - 1] = relative_T;
    // cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
    // cout << "init t_l " << T[l].transpose() << endl;

    // rotate to cam frame
    Eigen::Matrix3d c_Rotation[frame_num];
    Eigen::Vector3d c_Translation[frame_num];
    Eigen::Quaterniond c_Quat[frame_num];
    double c_rotation[frame_num][4];
    double c_translation[frame_num][3];
    Eigen::Matrix<double, 3, 4> Pose[frame_num];

    c_Quat[l] = q[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
    c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
    c_Translation[frame_num - 1] =
        -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
    Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
    Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

    // 1: trangulate between l ----- frame_num - 1
    // 2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
    for (int i = l; i < frame_num - 1; i++) {
        // solve pnp
        if (i > l) {
            Eigen::Matrix3d R_initial = c_Rotation[i - 1];
            Eigen::Vector3d P_initial = c_Translation[i - 1];
            if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
                return false;
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }

        // triangulate point based on the solve pnp result
        triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1],
                             sfm_f);
    }
    // 3: triangulate l-----l+1 l+2 ... frame_num -2
    for (int i = l + 1; i < frame_num - 1; i++)
        triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
    // 4: solve pnp l-1; triangulate l-1 ----- l
    //             l-2              l-2 ----- l
    for (int i = l - 1; i >= 0; i--) {
        // solve pnp
        Eigen::Matrix3d R_initial = c_Rotation[i + 1];
        Eigen::Vector3d P_initial = c_Translation[i + 1];
        if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
            return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        // triangulate
        triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
    }
    // 5: triangulate all other points
    for (int j = 0; j < feature_num; j++) {
        if (sfm_f[j].state == true)
            continue;
        if ((int)sfm_f[j].observation.size() >= 2) {
            Eigen::Vector2d point0, point1;
            int frame_0 = sfm_f[j].observation[0].first;
            point0 = sfm_f[j].observation[0].second;
            int frame_1 = sfm_f[j].observation.back().first;
            point1 = sfm_f[j].observation.back().second;
            Eigen::Vector3d point_3d;
            triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1,
                             point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
            // cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d
            // point : "  << j << "  " << point_3d.transpose() << endl;
        }
    }

    // full BA
    ceres::Problem problem;
    ceres::LocalParameterization *local_parameterization =
        new ceres::QuaternionParameterization();
    // cout << " begin full BA " << endl;
    for (int i = 0; i < frame_num; i++) {
        // double array for ceres
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();
        problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translation[i], 3);
        if (i == l) {
            problem.SetParameterBlockConstant(c_rotation[i]);
        }
        if (i == l || i == frame_num - 1) {
            problem.SetParameterBlockConstant(c_translation[i]);
        }
    }

    for (int i = 0; i < feature_num; i++) {
        if (sfm_f[i].state != true)
            continue;
        for (int j = 0; j < int(sfm_f[i].observation.size()); j++) {
            int l = sfm_f[i].observation[j].first;
            ceres::CostFunction *cost_function =
                ReprojectionError3D::Create(sfm_f[i].observation[j].second.x(),
                                            sfm_f[i].observation[j].second.y());

            problem.AddResidualBlock(cost_function, NULL, c_rotation[l],
                                     c_translation[l], sfm_f[i].position);
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << "\n";
    if (summary.termination_type == ceres::CONVERGENCE ||
        summary.final_cost < 5e-03) {
        // cout << "vision only BA converge" << endl;
    } else {
        // cout << "vision only BA not converge " << endl;
        return false;
    }
    for (int i = 0; i < frame_num; i++) {
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i] = q[i].inverse();
        // cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " <<
        // q[i].vec().transpose() << endl;
    }
    for (int i = 0; i < frame_num; i++) {
        T[i] = -1 *
               (q[i] * Eigen::Vector3d(c_translation[i][0], c_translation[i][1],
                                       c_translation[i][2]));
        // cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1)
        // <<"  "<< T[i](2) << endl;
    }
    for (int i = 0; i < (int)sfm_f.size(); i++) {
        if (sfm_f[i].state)
            sfm_tracked_points[sfm_f[i].id] =
                Eigen::Vector3d(sfm_f[i].position[0], sfm_f[i].position[1],
                                sfm_f[i].position[2]);
    }
    return true;
}

namespace cv {
void decomposeEssentialMat(InputArray _E, OutputArray _R1, OutputArray _R2,
                           OutputArray _t) {
    Mat E = _E.getMat().reshape(1, 3);
    CV_Assert(E.cols == 3 && E.rows == 3);

    Mat D, U, Vt;
    SVD::compute(E, D, U, Vt);

    if (determinant(U) < 0)
        U *= -1.;
    if (determinant(Vt) < 0)
        Vt *= -1.;

    Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
    W.convertTo(W, E.type());

    Mat R1, R2, t;
    R1 = U * W * Vt;
    R2 = U * W.t() * Vt;
    t = U.col(2) * 1.0;

    R1.copyTo(_R1);
    R2.copyTo(_R2);
    t.copyTo(_t);
}

int recoverPose(InputArray E, InputArray _points1, InputArray _points2,
                InputArray _cameraMatrix, OutputArray _R, OutputArray _t,
                InputOutputArray _mask) {
    Mat points1, points2, cameraMatrix;
    _points1.getMat().convertTo(points1, CV_64F);
    _points2.getMat().convertTo(points2, CV_64F);
    _cameraMatrix.getMat().convertTo(cameraMatrix, CV_64F);

    int npoints = points1.checkVector(2);
    CV_Assert(npoints >= 0 && points2.checkVector(2) == npoints &&
              points1.type() == points2.type());

    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 &&
              cameraMatrix.channels() == 1);

    if (points1.channels() > 1) {
        points1 = points1.reshape(1, npoints);
        points2 = points2.reshape(1, npoints);
    }

    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    points1.col(0) = (points1.col(0) - cx) / fx;
    points2.col(0) = (points2.col(0) - cx) / fx;
    points1.col(1) = (points1.col(1) - cy) / fy;
    points2.col(1) = (points2.col(1) - cy) / fy;

    points1 = points1.t();
    points2 = points2.t();

    Mat R1, R2, t;
    decomposeEssentialMat(E, R1, R2, t);
    Mat P0 = Mat::eye(3, 4, R1.type());
    Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()),
        P4(3, 4, R1.type());
    P1(Range::all(), Range(0, 3)) = R1 * 1.0;
    P1.col(3) = t * 1.0;
    P2(Range::all(), Range(0, 3)) = R2 * 1.0;
    P2.col(3) = t * 1.0;
    P3(Range::all(), Range(0, 3)) = R1 * 1.0;
    P3.col(3) = -t * 1.0;
    P4(Range::all(), Range(0, 3)) = R2 * 1.0;
    P4.col(3) = -t * 1.0;

    // Do the cheirality check.
    // Notice here a threshold dist is used to filter
    // out far away points (i.e. infinite points) since
    // there depth may vary between postive and negtive.
    double dist = 50.0;
    Mat Q;
    triangulatePoints(P0, P1, points1, points2, Q);
    Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask1 = (Q.row(2) < dist) & mask1;
    Q = P1 * Q;
    mask1 = (Q.row(2) > 0) & mask1;
    mask1 = (Q.row(2) < dist) & mask1;

    triangulatePoints(P0, P2, points1, points2, Q);
    Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask2 = (Q.row(2) < dist) & mask2;
    Q = P2 * Q;
    mask2 = (Q.row(2) > 0) & mask2;
    mask2 = (Q.row(2) < dist) & mask2;

    triangulatePoints(P0, P3, points1, points2, Q);
    Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask3 = (Q.row(2) < dist) & mask3;
    Q = P3 * Q;
    mask3 = (Q.row(2) > 0) & mask3;
    mask3 = (Q.row(2) < dist) & mask3;

    triangulatePoints(P0, P4, points1, points2, Q);
    Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask4 = (Q.row(2) < dist) & mask4;
    Q = P4 * Q;
    mask4 = (Q.row(2) > 0) & mask4;
    mask4 = (Q.row(2) < dist) & mask4;

    mask1 = mask1.t();
    mask2 = mask2.t();
    mask3 = mask3.t();
    mask4 = mask4.t();

    // If _mask is given, then use it to filter outliers.
    if (!_mask.empty()) {
        Mat mask = _mask.getMat();
        CV_Assert(mask.size() == mask1.size());
        bitwise_and(mask, mask1, mask1);
        bitwise_and(mask, mask2, mask2);
        bitwise_and(mask, mask3, mask3);
        bitwise_and(mask, mask4, mask4);
    }
    if (_mask.empty() && _mask.needed()) {
        _mask.create(mask1.size(), CV_8U);
    }

    CV_Assert(_R.needed() && _t.needed());
    _R.create(3, 3, R1.type());
    _t.create(3, 1, t.type());

    int good1 = countNonZero(mask1);
    int good2 = countNonZero(mask2);
    int good3 = countNonZero(mask3);
    int good4 = countNonZero(mask4);

    if (good1 >= good2 && good1 >= good3 && good1 >= good4) {
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask1.copyTo(_mask);
        return good1;
    } else if (good2 >= good1 && good2 >= good3 && good2 >= good4) {
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask2.copyTo(_mask);
        return good2;
    } else if (good3 >= good1 && good3 >= good2 && good3 >= good4) {
        t = -t;
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask3.copyTo(_mask);
        return good3;
    } else {
        t = -t;
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed())
            mask4.copyTo(_mask);
        return good4;
    }
}

int recoverPose(InputArray E, InputArray _points1, InputArray _points2,
                OutputArray _R, OutputArray _t, double focal, Point2d pp,
                InputOutputArray _mask) {
    Mat cameraMatrix =
        (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
    return cv::recoverPose(E, _points1, _points2, cameraMatrix, _R, _t, _mask);
}
}  // namespace cv

bool solveRelativeRT(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres,
    Eigen::Matrix3d &Rotation, Eigen::Vector3d &Translation) {
    if (corres.size() >= 15) {
        std::vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++) {
            ll.push_back(cv::Point2f(corres[i].first(0) / corres[i].first(2),
                                     corres[i].first(1) / corres[i].first(2)));
            rr.push_back(
                cv::Point2f(corres[i].second(0) / corres[i].second(2),
                            corres[i].second(1) / corres[i].second(2)));
        }
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460,
                                           0.99, mask);
        cv::Mat cameraMatrix =
            (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        int inlier_cnt =
            cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        // cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++) {
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        if (inlier_cnt > 12)
            return true;
        else
            return false;
    }
    return false;
}
