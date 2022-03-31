#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "sophus/so3.hpp"

/** \brief Ceres Cost Funtion of VGICP
 */
struct Cost_IMU_VGICP {
    Cost_IMU_VGICP(const int &_N, const Eigen::Vector4d &_mean_b,
                   const Eigen::Vector4d &_a, const Eigen::Matrix4d &_mean_CB,
                   const Eigen::Matrix4d &_CA,
                   const Eigen::Matrix<double, 1, 1> _sqrt_information)
        : N(_N),
          mean_b_(_mean_b),
          a_(_a),
          mean_CB_(_mean_CB),
          CA_(_CA),
          sqrt_information(_sqrt_information) {}

    template <typename T>
    bool operator()(const T *PRi, const T *Exbl, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
        Eigen::Matrix<T, 4, 4> T_wb = Eigen::Matrix<T, 4, 4>::Identity();
        T_wb.template block<3, 3>(0, 0) =
            Sophus::SO3<T>::exp(pri_wb.template segment<3>(3)).matrix();
        T_wb.template block<3, 1>(0, 3) = pri_wb.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_bl(Exbl);
        Eigen::Matrix<T, 4, 4> T_bl = Eigen::Matrix<T, 4, 4>::Identity();
        T_bl.template block<3, 3>(0, 0) =
            Sophus::SO3<T>::exp(pri_bl.template segment<3>(3)).matrix();
        T_bl.template block<3, 1>(0, 3) = pri_bl.template segment<3>(0);

        Eigen::Matrix<T, 4, 4> Twl = T_wb * T_bl;
        Eigen::Matrix<T, 4, 1> a = a_.cast<T>();
        Eigen::Matrix<T, 4, 1> pw = Twl * a;
        Eigen::Matrix<T, 4, 1> mean_b = mean_b_.cast<T>();
        Eigen::Matrix<T, 4, 1> AA = mean_b - pw;

        Eigen::Matrix<T, 4, 4> mean_CB = mean_CB_.cast<T>();
        Eigen::Matrix<T, 4, 4> CA = CA_.cast<T>();

        Eigen::Matrix<T, 4, 4> BB = mean_CB + Twl * CA * Twl.transpose();

        Eigen::Matrix<T, 1, 1> re = T(N) * AA.transpose() * BB.inverse() * AA;
        T residual_ = re(0);
        residual[0] = T(sqrt_information(0)) * residual_;

        return true;
    }

    static ceres::CostFunction *Create(
        const int &_N, const Eigen::Vector4d &_mean_b,
        const Eigen::Vector4d &_a, const Eigen::Matrix4d &_mean_CB,
        const Eigen::Matrix4d &_CA,
        const Eigen::Matrix<double, 1, 1> &_sqrt_information) {
        return (new ceres::AutoDiffCostFunction<Cost_IMU_VGICP, 1, 6, 6>(
            new Cost_IMU_VGICP(_N, _mean_b, _a, _mean_CB, _CA,
                               _sqrt_information)));
    }

    int N;
    Eigen::Vector4d mean_b_;
    Eigen::Vector4d a_;
    Eigen::Matrix4d mean_CB_;
    Eigen::Matrix4d CA_;
    Eigen::Matrix<double, 1, 1> sqrt_information;
};

/** \brief Ceres Cost Funtion of VGICP
 */
struct Cost_IMU_VGICP2 {
    Cost_IMU_VGICP2(const int &_N, const Eigen::Vector4d &_mean_b,
                    const Eigen::Vector4d &_a, const Eigen::Matrix4d &_BB_inv,
                    const Eigen::Matrix<double, 1, 1> _sqrt_information)
        : N(_N),
          mean_b_(_mean_b),
          a_(_a),
          BB_inv_(_BB_inv),
          sqrt_information(_sqrt_information) {}

    template <typename T>
    bool operator()(const T *PRi, const T *Exbl, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
        Eigen::Matrix<T, 4, 4> T_wb = Eigen::Matrix<T, 4, 4>::Identity();
        T_wb.template block<3, 3>(0, 0) =
            Sophus::SO3<T>::exp(pri_wb.template segment<3>(3)).matrix();
        T_wb.template block<3, 1>(0, 3) = pri_wb.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_bl(Exbl);
        Eigen::Matrix<T, 4, 4> T_bl = Eigen::Matrix<T, 4, 4>::Identity();
        T_bl.template block<3, 3>(0, 0) =
            Sophus::SO3<T>::exp(pri_bl.template segment<3>(3)).matrix();
        T_bl.template block<3, 1>(0, 3) = pri_bl.template segment<3>(0);

        Eigen::Matrix<T, 4, 4> Twl = T_wb * T_bl;
        Eigen::Matrix<T, 4, 1> a = a_.cast<T>();
        Eigen::Matrix<T, 4, 1> pw = Twl * a;
        Eigen::Matrix<T, 4, 1> mean_b = mean_b_.cast<T>();
        Eigen::Matrix<T, 4, 1> AA = mean_b - pw;

        Eigen::Matrix<T, 4, 4> BB_inv = BB_inv_.cast<T>();

        Eigen::Matrix<T, 1, 1> re = T(N) * AA.transpose() * BB_inv * AA;
        T residual_ = re(0);
        residual[0] = T(sqrt_information(0)) * residual_;

        return true;
    }

    static ceres::CostFunction *Create(
        const int &_N, const Eigen::Vector4d &_mean_b,
        const Eigen::Vector4d &_a, const Eigen::Matrix4d &_BB_inv,
        const Eigen::Matrix<double, 1, 1> &_sqrt_information) {
        return (new ceres::AutoDiffCostFunction<Cost_IMU_VGICP2, 1, 6, 6>(
            new Cost_IMU_VGICP2(_N, _mean_b, _a, _BB_inv, _sqrt_information)));
    }

    int N;
    Eigen::Vector4d mean_b_;
    Eigen::Vector4d a_;
    Eigen::Matrix4d BB_inv_;
    Eigen::Matrix<double, 1, 1> sqrt_information;
};