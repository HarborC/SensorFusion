#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "sophus/so3.hpp"

// Eigen::Vector3d sqrt_information_vec(10, 10, 10); // close
Eigen::Vector3d sqrt_information_vec(0.0000205, 0.0000205, 0.0001);
Eigen::Matrix<double, 3, 3> Cost_NavState_PR_Ground::sqrt_information =
    sqrt_information_vec.asDiagonal().inverse();
Eigen::VectorXf Cost_NavState_PR_Ground::init_ground_plane_coeff(4);

/** \brief Ceres Cost Funtion between Lidar Pose and Ground
 */
struct Cost_NavState_PR_Ground {
    Cost_NavState_PR_Ground(Eigen::VectorXf ground_plane_coeff_)
        : ground_plane_coeff(ground_plane_coeff_) {
        // std::cout << ground_plane_coeff.transpose() << std::endl;
    }

    template <typename T>
    bool operator()(const T *pri_, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> PRi(pri_);
        Eigen::Matrix<T, 3, 1> Pi = PRi.template segment<3>(0);
        Sophus::SO3<T> SO3_Ri = Sophus::SO3<T>::exp(PRi.template segment<3>(3));
        Eigen::Map<Eigen::Matrix<T, 3, 1>> eResiduals(residual);
        eResiduals = Eigen::Matrix<T, 3, 1>::Zero();

        Eigen::Matrix<T, 4, 4> T_wl = Eigen::Matrix<T, 4, 4>::Identity();
        T_wl.topLeftCorner(3, 3) = SO3_Ri.matrix();
        T_wl.topRightCorner(3, 1) = Pi;
        Eigen::Matrix<T, 4, 4> T_lw = T_wl.inverse();
        Eigen::Matrix<T, 3, 3> R_lw = T_lw.topLeftCorner(3, 3);
        Eigen::Matrix<T, 3, 1> t_lw = T_lw.topRightCorner(3, 1);

        Eigen::Matrix<T, 4, 1> ground_plane_coeff_temp =
            ground_plane_coeff.cast<T>().template segment<4>(0);
        Eigen::Matrix<T, 4, 1> local_ground_plane;
        local_ground_plane.template segment<3>(0) =
            R_lw * init_ground_plane_coeff.cast<T>().template segment<3>(0);
        local_ground_plane.template segment<1>(3) =
            init_ground_plane_coeff.cast<T>().template segment<1>(3) -
            t_lw.transpose() * local_ground_plane.template segment<3>(0);
        eResiduals =
            SensorFusion::ominus(ground_plane_coeff_temp, local_ground_plane);
        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(Eigen::VectorXf ground_plane_coeff) {
        return (new ceres::AutoDiffCostFunction<Cost_NavState_PR_Ground, 3, 6>(
            new Cost_NavState_PR_Ground(ground_plane_coeff)));
    }
    Eigen::VectorXf ground_plane_coeff;
    static Eigen::Matrix<double, 3, 3> sqrt_information;
    static Eigen::VectorXf init_ground_plane_coeff;
};
