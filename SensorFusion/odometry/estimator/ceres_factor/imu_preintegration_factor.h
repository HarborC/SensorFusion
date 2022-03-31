#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "sophus/so3.hpp"

#include "../../imu_integrator.h"

/** \brief Ceres Cost Funtion between Lidar Pose and IMU Preintegration
 */
struct Cost_NavState_PRV_Bias {
    Cost_NavState_PRV_Bias(SensorFusion::IMUIntegrator &measure_,
                           Eigen::Vector3d &GravityVec_,
                           Eigen::Matrix<double, 15, 15> sqrt_information_)
        : imu_measure(measure_),
          GravityVec(GravityVec_),
          sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *pri_, const T *velobiasi_, const T *prj_,
                    const T *velobiasj_, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> PRi(pri_);
        Eigen::Matrix<T, 3, 1> Pi = PRi.template segment<3>(0);
        Sophus::SO3<T> SO3_Ri = Sophus::SO3<T>::exp(PRi.template segment<3>(3));

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> PRj(prj_);
        Eigen::Matrix<T, 3, 1> Pj = PRj.template segment<3>(0);
        Sophus::SO3<T> SO3_Rj = Sophus::SO3<T>::exp(PRj.template segment<3>(3));

        Eigen::Map<const Eigen::Matrix<T, 9, 1>> velobiasi(velobiasi_);
        Eigen::Matrix<T, 3, 1> Vi = velobiasi.template segment<3>(0);
        Eigen::Matrix<T, 3, 1> dbgi = velobiasi.template segment<3>(3) -
                                      imu_measure.GetBiasGyr().cast<T>();
        Eigen::Matrix<T, 3, 1> dbai = velobiasi.template segment<3>(6) -
                                      imu_measure.GetBiasAcc().cast<T>();

        Eigen::Map<const Eigen::Matrix<T, 9, 1>> velobiasj(velobiasj_);
        Eigen::Matrix<T, 3, 1> Vj = velobiasj.template segment<3>(0);

        Eigen::Map<Eigen::Matrix<T, 15, 1>> eResiduals(residual);
        eResiduals = Eigen::Matrix<T, 15, 1>::Zero();

        T dTij = T(imu_measure.GetDeltaTime());
        T dT2 = dTij * dTij;
        Eigen::Matrix<T, 3, 1> dPij = imu_measure.GetDeltaP().cast<T>();
        Eigen::Matrix<T, 3, 1> dVij = imu_measure.GetDeltaV().cast<T>();
        Sophus::SO3<T> dRij = Sophus::SO3<T>(imu_measure.GetDeltaQ().cast<T>());
        Sophus::SO3<T> RiT = SO3_Ri.inverse();

        Eigen::Matrix<T, 3, 1> rPij =
            RiT * (Pj - Pi - Vi * dTij - 0.5 * GravityVec.cast<T>() * dT2) -
            (dPij +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_P,
                                  SensorFusion::IMUIntegrator::O_BG)
                     .cast<T>() *
                 dbgi +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_P,
                                  SensorFusion::IMUIntegrator::O_BA)
                     .cast<T>() *
                 dbai);

        Sophus::SO3<T> dR_dbg = Sophus::SO3<T>::exp(
            imu_measure.GetJacobian()
                .block<3, 3>(SensorFusion::IMUIntegrator::O_R,
                             SensorFusion::IMUIntegrator::O_BG)
                .cast<T>() *
            dbgi);
        Sophus::SO3<T> rRij = (dRij * dR_dbg).inverse() * RiT * SO3_Rj;
        Eigen::Matrix<T, 3, 1> rPhiij = rRij.log();

        Eigen::Matrix<T, 3, 1> rVij =
            RiT * (Vj - Vi - GravityVec.cast<T>() * dTij) -
            (dVij +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_V,
                                  SensorFusion::IMUIntegrator::O_BG)
                     .cast<T>() *
                 dbgi +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_V,
                                  SensorFusion::IMUIntegrator::O_BA)
                     .cast<T>() *
                 dbai);

        eResiduals.template segment<3>(0) = rPij;
        eResiduals.template segment<3>(3) = rPhiij;
        eResiduals.template segment<3>(6) = rVij;
        eResiduals.template segment<6>(9) =
            velobiasj.template segment<6>(3) - velobiasi.template segment<6>(3);

        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(
        SensorFusion::IMUIntegrator &measure_, Eigen::Vector3d &GravityVec_,
        Eigen::Matrix<double, 15, 15> sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_NavState_PRV_Bias, 15, 6, 9, 6,
                                            9>(new Cost_NavState_PRV_Bias(
                measure_, GravityVec_, std::move(sqrt_information_))));
    }

    SensorFusion::IMUIntegrator imu_measure;
    Eigen::Vector3d GravityVec;
    Eigen::Matrix<double, 15, 15> sqrt_information;
};
