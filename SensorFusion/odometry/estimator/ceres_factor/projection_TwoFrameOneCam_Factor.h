#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "sophus/so3.hpp"

struct ProjectionTwoFrameOneCamFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionTwoFrameOneCamFactor(
        const Eigen::Vector3d &pts_i_, const Eigen::Vector3d &pts_j_,
        Eigen::Matrix<double, 2, 2> sqrt_information_)
        : pts_i(pts_i_),
          pts_j(pts_j_),
          sqrt_information(std::move(sqrt_information_)) {
        Eigen::Vector3d b1, b2;
        Eigen::Vector3d a = pts_j.normalized();
        Eigen::Vector3d tmp(0, 0, 1);
        if (a == tmp)
            tmp << 1, 0, 0;
        b1 = (tmp - a * (a.transpose() * tmp)).normalized();
        b2 = a.cross(b1);
        tangent_base.block<1, 3>(0, 0) = b1.transpose();
        tangent_base.block<1, 3>(1, 0) = b2.transpose();
    }

    template <typename T>
    bool operator()(const T *PRi, const T *PRj, const T *Exbc, const T *invDep,
                    T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wbi(PRi);
        Eigen::Quaternion<T> q_wbi =
            Sophus::SO3<T>::exp(pri_wbi.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wbi = pri_wbi.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wbj(PRj);
        Eigen::Quaternion<T> q_wbj =
            Sophus::SO3<T>::exp(pri_wbj.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wbj = pri_wbj.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> ex_ic(Exbc);
        Eigen::Quaternion<T> qic =
            Sophus::SO3<T>::exp(ex_ic.template segment<3>(3)).unit_quaternion();
        Eigen::Matrix<T, 3, 1> tic = ex_ic.template segment<3>(0);

        T inv_dep_i = invDep[0];

        Eigen::Matrix<T, 3, 1> pts_i_td, pts_j_td;
        pts_i_td = pts_i.cast<T>();
        pts_j_td = pts_j.cast<T>();

        Eigen::Matrix<T, 3, 1> pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Matrix<T, 3, 1> pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Matrix<T, 3, 1> pts_w = q_wbi * pts_imu_i + t_wbi;
        Eigen::Matrix<T, 3, 1> pts_imu_j = q_wbj.inverse() * (pts_w - t_wbj);
        Eigen::Matrix<T, 3, 1> pts_camera_j = qic.inverse() * (pts_imu_j - tic);

        Eigen::Map<Eigen::Matrix<T, 2, 1>> eResiduals(residual);
        eResiduals = tangent_base.cast<T>() *
                     (pts_camera_j.normalized() - pts_j_td.normalized());
        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &pts_i_, const Eigen::Vector3d &pts_j_,
        Eigen::Matrix<double, 2, 2> sqrt_information_) {
        return (new ceres::AutoDiffCostFunction<ProjectionTwoFrameOneCamFactor,
                                                2, 6, 6, 6, 1>(
            new ProjectionTwoFrameOneCamFactor(pts_i_, pts_j_,
                                               sqrt_information_)));
    }

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    Eigen::Matrix<double, 2, 2> sqrt_information;
};
