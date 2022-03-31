#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "sophus/so3.hpp"

struct PriorFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PriorFactor(const Eigen::Matrix4d &pose) {
        q_prior_ = Sophus::SO3<double>(pose.block<3, 3>(0, 0));
        t_prior_ = pose.block<3, 1>(0, 3);
    }

    template <typename T>
    bool operator()(const T *Ex_Para, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> Ex_para(Ex_Para);
        Sophus::SO3<T> q_para =
            Sophus::SO3<T>::exp(Ex_para.template segment<3>(3));
        Eigen::Matrix<T, 3, 1> t_para = Ex_para.template segment<3>(0);

        Sophus::SO3<T> q_prior = q_prior_.cast<T>();
        Eigen::Matrix<T, 3, 1> t_prior = t_prior_.cast<T>();

        Sophus::SO3<T> d_R = q_prior.inverse() * q_para;
        Eigen::Matrix<T, 3, 1> d_r = d_R.log();
        Eigen::Matrix<T, 3, 1> d_t = t_para - t_prior;

        T info_t = T(1000);
        T info_r = T(1);

        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residual);
        residuals[0] = info_t * d_t(0);
        residuals[1] = info_t * d_t(1);
        residuals[2] = info_t * d_t(2);
        residuals[3] = info_r * d_r(0);
        residuals[4] = info_r * d_r(1);
        residuals[5] = info_r * d_r(2);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Matrix4d &pose) {
        return (new ceres::AutoDiffCostFunction<PriorFactor, 6, 6>(
            new PriorFactor(pose)));
    }

    Sophus::SO3<double> q_prior_;
    Eigen::Matrix<double, 3, 1> t_prior_;
};