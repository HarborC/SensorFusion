#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "sophus/so3.hpp"

/** \brief Ceres Cost Funtion between PointCloud Flat Feature and Map Cloud
 */
struct Cost_NavState_IMU_Plan {
    Cost_NavState_IMU_Plan(Eigen::Vector3d _p, double _pa, double _pb,
                           double _pc, double _pd,
                           Eigen::Matrix<double, 1, 1> sqrt_information_)
        : point(std::move(_p)),
          pa(_pa),
          pb(_pb),
          pc(_pc),
          pd(_pd),
          sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *PRi, const T *Exbl, T *residual) const {
        Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
        Eigen::Quaternion<T> q_wb =
            Sophus::SO3<T>::exp(pri_wb.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wb = pri_wb.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_bl(Exbl);
        Eigen::Quaternion<T> qbl =
            Sophus::SO3<T>::exp(pri_bl.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> Pbl = pri_bl.template segment<3>(0);

        Eigen::Quaternion<T> q_wl = q_wb * qbl;
        Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl + t_wb;
        Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

        T pd2 = T(pa) * P_to_Map(0) + T(pb) * P_to_Map(1) +
                T(pc) * P_to_Map(2) + T(pd);
        T _weight =
            T(1) - T(0.9) * ceres::abs(pd2) /
                       ceres::sqrt(ceres::sqrt(P_to_Map(0) * P_to_Map(0) +
                                               P_to_Map(1) * P_to_Map(1) +
                                               P_to_Map(2) * P_to_Map(2)));
        residual[0] = T(sqrt_information(0)) * _weight * pd2;
        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &curr_point_, const double &pa_,
        const double &pb_, const double &pc_, const double &pd_,
        Eigen::Matrix<double, 1, 1> sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_NavState_IMU_Plan, 1, 6, 6>(
                new Cost_NavState_IMU_Plan(curr_point_, pa_, pb_, pc_, pd_,
                                           std::move(sqrt_information_))));
    }

    double pa, pb, pc, pd;
    Eigen::Vector3d point;
    Eigen::Matrix<double, 1, 1> sqrt_information;
};

/** \brief Ceres Cost Funtion between PointCloud Flat Feature and Map Cloud
 */
struct Cost_NavState_IMU_Plan_Vec {
    Cost_NavState_IMU_Plan_Vec(Eigen::Vector3d _p, Eigen::Vector3d _p_proj,
                               Eigen::Matrix<double, 3, 3> _sqrt_information)
        : point(std::move(_p)),
          point_proj(std::move(_p_proj)),
          sqrt_information(std::move(_sqrt_information)) {}

    template <typename T>
    bool operator()(const T *PRi, const T *Exbl, T *residual) const {
        Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};
        Eigen::Matrix<T, 3, 1> cp_proj{T(point_proj.x()), T(point_proj.y()),
                                       T(point_proj.z())};

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
        Eigen::Quaternion<T> q_wb =
            Sophus::SO3<T>::exp(pri_wb.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wb = pri_wb.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_bl(Exbl);
        Eigen::Quaternion<T> qbl =
            Sophus::SO3<T>::exp(pri_bl.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> Pbl = pri_bl.template segment<3>(0);

        Eigen::Quaternion<T> q_wl = q_wb * qbl;
        Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl + t_wb;
        Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

        Eigen::Map<Eigen::Matrix<T, 3, 1>> eResiduals(residual);
        eResiduals = P_to_Map - cp_proj;
        T _weight =
            T(1) - T(0.9) * (P_to_Map - cp_proj).norm() /
                       ceres::sqrt(ceres::sqrt(P_to_Map(0) * P_to_Map(0) +
                                               P_to_Map(1) * P_to_Map(1) +
                                               P_to_Map(2) * P_to_Map(2)));
        eResiduals *= _weight;
        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &p_proj_,
        const Eigen::Matrix<double, 3, 3> sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_NavState_IMU_Plan_Vec, 3, 6,
                                            6>(new Cost_NavState_IMU_Plan_Vec(
                curr_point_, p_proj_, sqrt_information_)));
    }

    Eigen::Vector3d point;
    Eigen::Vector3d point_proj;
    Eigen::Matrix<double, 3, 3> sqrt_information;
};
