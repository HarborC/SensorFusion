#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "sophus/so3.hpp"

/** \brief Ceres Cost Funtion between PointCloud Sharp Feature and Map Cloud
 */
struct Cost_NavState_IMU_Line {
    Cost_NavState_IMU_Line(Eigen::Vector3d _p, Eigen::Vector3d _vtx1,
                           Eigen::Vector3d _vtx2,
                           Eigen::Matrix<double, 1, 1> sqrt_information_)
        : point(std::move(_p)),
          vtx1(std::move(_vtx1)),
          vtx2(std::move(_vtx2)),
          sqrt_information(std::move(sqrt_information_)) {
        l12 = std::sqrt((vtx1(0) - vtx2(0)) * (vtx1(0) - vtx2(0)) +
                        (vtx1(1) - vtx2(1)) * (vtx1(1) - vtx2(1)) +
                        (vtx1(2) - vtx2(2)) * (vtx1(2) - vtx2(2)));
    }

    template <typename T>
    bool operator()(const T *PRi, const T *Exbl, T *residual) const {
        Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};
        Eigen::Matrix<T, 3, 1> lpa{T(vtx1.x()), T(vtx1.y()), T(vtx1.z())};
        Eigen::Matrix<T, 3, 1> lpb{T(vtx2.x()), T(vtx2.y()), T(vtx2.z())};

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

        T a012 =
            ceres::sqrt(((P_to_Map(0) - lpa(0)) * (P_to_Map(1) - lpb(1)) -
                         (P_to_Map(0) - lpb(0)) * (P_to_Map(1) - lpa(1))) *
                            ((P_to_Map(0) - lpa(0)) * (P_to_Map(1) - lpb(1)) -
                             (P_to_Map(0) - lpb(0)) * (P_to_Map(1) - lpa(1))) +
                        ((P_to_Map(0) - lpa(0)) * (P_to_Map(2) - lpb(2)) -
                         (P_to_Map(0) - lpb(0)) * (P_to_Map(2) - lpa(2))) *
                            ((P_to_Map(0) - lpa(0)) * (P_to_Map(2) - lpb(2)) -
                             (P_to_Map(0) - lpb(0)) * (P_to_Map(2) - lpa(2))) +
                        ((P_to_Map(1) - lpa(1)) * (P_to_Map(2) - lpb(2)) -
                         (P_to_Map(1) - lpb(1)) * (P_to_Map(2) - lpa(2))) *
                            ((P_to_Map(1) - lpa(1)) * (P_to_Map(2) - lpb(2)) -
                             (P_to_Map(1) - lpb(1)) * (P_to_Map(2) - lpa(2))));
        T ld2 = a012 / T(l12);
        T _weight =
            T(1) - T(0.9) * ceres::abs(ld2) /
                       ceres::sqrt(ceres::sqrt(P_to_Map(0) * P_to_Map(0) +
                                               P_to_Map(1) * P_to_Map(1) +
                                               P_to_Map(2) * P_to_Map(2)));
        residual[0] = T(sqrt_information(0)) * _weight * ld2;
        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &curr_point_,
        const Eigen::Vector3d &last_point_a_,
        const Eigen::Vector3d &last_point_b_,
        Eigen::Matrix<double, 1, 1> sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_NavState_IMU_Line, 1, 6, 6>(
                new Cost_NavState_IMU_Line(curr_point_, last_point_a_,
                                           last_point_b_,
                                           std::move(sqrt_information_))));
    }

    Eigen::Vector3d point;
    Eigen::Vector3d vtx1;
    Eigen::Vector3d vtx2;
    double l12;
    Eigen::Matrix<double, 1, 1> sqrt_information;
};
