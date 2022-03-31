#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "sophus/so3.hpp"

/** \brief Ceres Cost Funtion for Initial Gravity Direction
 */
struct Cost_Initial_G {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Cost_Initial_G(Eigen::Vector3d acc_) : acc(acc_) {}

    template <typename T>
    bool operator()(const T *q, T *residual) const {
        Eigen::Matrix<T, 3, 1> acc_T = acc.cast<T>();
        Eigen::Quaternion<T> q_wg{q[0], q[1], q[2], q[3]};
        Eigen::Matrix<T, 3, 1> g_I{T(0), T(0), T(-9.805)};
        Eigen::Matrix<T, 3, 1> resi = q_wg * g_I - acc_T;
        residual[0] = resi[0];
        residual[1] = resi[1];
        residual[2] = resi[2];

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d acc_) {
        return (new ceres::AutoDiffCostFunction<Cost_Initial_G, 3, 4>(
            new Cost_Initial_G(acc_)));
    }

    Eigen::Vector3d acc;
};

/** \brief Ceres Cost Funtion of IMU Factor in LIO Initialization
 */
struct Cost_Initialization_IMU {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Cost_Initialization_IMU(SensorFusion::IMUIntegrator &measure_,
                            Eigen::Matrix<double, 9, 9> sqrt_information_)
        : imu_measure(measure_),
          sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *Ti_, const T *Tj_, const T *Tcb_, const T *vi_,
                    const T *vj_, const T *ba_, const T *bg_, const T *rwg_,
                    const T *s_, T *residual) const {
        Eigen::Matrix<T, 3, 1> G_I{T(0), T(0), T(-9.805)};

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba(ba_);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg(bg_);
        Eigen::Matrix<T, 3, 1> dbg = bg - imu_measure.GetBiasGyr().cast<T>();
        Eigen::Matrix<T, 3, 1> dba = ba - imu_measure.GetBiasAcc().cast<T>();

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> Tcb(Tcb_);
        Eigen::Matrix<T, 3, 1> tcb = Tcb.template segment<3>(0);
        Sophus::SO3<T> SO3_Rcb =
            Sophus::SO3<T>::exp(Tcb.template segment<3>(3));

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> Ti(Ti_);
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> Tj(Tj_);
        Eigen::Matrix<T, 3, 1> ti = Ti.template segment<3>(0);
        Eigen::Matrix<T, 3, 1> tj = Tj.template segment<3>(0);

        Sophus::SO3<T> SO3_Ri =
            Sophus::SO3<T>::exp(Ti.template segment<3>(3)) * SO3_Rcb;
        Sophus::SO3<T> SO3_RiT = SO3_Ri.inverse();
        Sophus::SO3<T> SO3_Rj =
            Sophus::SO3<T>::exp(Tj.template segment<3>(0)) * SO3_Rcb;

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> rwg(rwg_);
        Sophus::SO3<T> SO3_Rwg = Sophus::SO3<T>::exp(rwg);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> vi(vi_);
        Eigen::Matrix<T, 3, 1> Vi = vi;
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> vj(vj_);
        Eigen::Matrix<T, 3, 1> Vj = vj;

        T scale = s_[0];

        Eigen::Map<Eigen::Matrix<T, 9, 1>> eResiduals(residual);
        eResiduals = Eigen::Matrix<T, 9, 1>::Zero();

        T dTij = T(imu_measure.GetDeltaTime());
        T dT2 = dTij * dTij;
        Eigen::Matrix<T, 3, 1> dPij = imu_measure.GetDeltaP().cast<T>();
        Eigen::Matrix<T, 3, 1> dVij = imu_measure.GetDeltaV().cast<T>();
        Sophus::SO3<T> dRij = Sophus::SO3<T>(imu_measure.GetDeltaQ().cast<T>());

        Eigen::Matrix<T, 3, 1> rPij =
            SO3_RiT *
                (SO3_Rj * tcb - SO3_Ri * tcb + scale * (tj - ti - Vi * dTij) -
                 SO3_Rwg * G_I * dT2 * T(0.5)) -
            (dPij +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_P,
                                  SensorFusion::IMUIntegrator::O_BG)
                     .cast<T>() *
                 dbg +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_P,
                                  SensorFusion::IMUIntegrator::O_BA)
                     .cast<T>() *
                 dba);

        Sophus::SO3<T> dR_dbg = Sophus::SO3<T>::exp(
            imu_measure.GetJacobian()
                .block<3, 3>(SensorFusion::IMUIntegrator::O_R,
                             SensorFusion::IMUIntegrator::O_BG)
                .cast<T>() *
            dbg);
        Sophus::SO3<T> rRij = (dRij * dR_dbg).inverse() * SO3_RiT * SO3_Rj;
        Eigen::Matrix<T, 3, 1> rPhiij = rRij.log();

        Eigen::Matrix<T, 3, 1> rVij =
            SO3_RiT * (scale * (Vj - Vi) - SO3_Rwg * G_I * dTij) -
            (dVij +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_V,
                                  SensorFusion::IMUIntegrator::O_BG)
                     .cast<T>() *
                 dbg +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_V,
                                  SensorFusion::IMUIntegrator::O_BA)
                     .cast<T>() *
                 dba);

        eResiduals.template segment<3>(0) = rPij;
        eResiduals.template segment<3>(3) = rPhiij;
        eResiduals.template segment<3>(6) = rVij;

        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(
        SensorFusion::IMUIntegrator &measure_,
        Eigen::Matrix<double, 9, 9> sqrt_information_) {
        return (new ceres::AutoDiffCostFunction<Cost_Initialization_IMU, 9, 6,
                                                6, 6, 3, 3, 3, 3, 3, 1>(
            new Cost_Initialization_IMU(measure_,
                                        std::move(sqrt_information_))));
    }

    SensorFusion::IMUIntegrator imu_measure;
    Eigen::Matrix<double, 9, 9> sqrt_information;
};

/** \brief Ceres Cost Funtion of IMU Biases and Velocity Prior Factor in LIO
 * Initialization
 */
struct Cost_Initialization_Prior_bv {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Cost_Initialization_Prior_bv(Eigen::Vector3d prior_,
                                 Eigen::Matrix3d sqrt_information_)
        : prior(prior_), sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *bv_, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bv(bv_);
        Eigen::Matrix<T, 3, 1> Bv = bv;

        Eigen::Matrix<T, 3, 1> prior_T(prior.cast<T>());
        Eigen::Matrix<T, 3, 1> prior_Bv = prior_T;

        Eigen::Map<Eigen::Matrix<T, 3, 1>> eResiduals(residual);
        eResiduals = Eigen::Matrix<T, 3, 1>::Zero();

        eResiduals = Bv - prior_Bv;

        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d prior_,
                                       Eigen::Matrix3d sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_Initialization_Prior_bv, 3, 3>(
                new Cost_Initialization_Prior_bv(
                    prior_, std::move(sqrt_information_))));
    }

    Eigen::Vector3d prior;
    Eigen::Matrix3d sqrt_information;
};

/** \brief Ceres Cost Funtion of Rwg Prior Factor in LIO Initialization
 */
struct Cost_Initialization_Prior_R {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Cost_Initialization_Prior_R(Eigen::Vector3d prior_,
                                Eigen::Matrix3d sqrt_information_)
        : prior(prior_), sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *r_wg_, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> r_wg(r_wg_);
        Eigen::Matrix<T, 3, 1> R_wg = r_wg;
        Sophus::SO3<T> SO3_R_wg = Sophus::SO3<T>::exp(R_wg);

        Eigen::Matrix<T, 3, 1> prior_T(prior.cast<T>());
        Sophus::SO3<T> prior_R_wg = Sophus::SO3<T>::exp(prior_T);

        Sophus::SO3<T> d_R = SO3_R_wg.inverse() * prior_R_wg;
        Eigen::Matrix<T, 3, 1> d_Phi = d_R.log();

        Eigen::Map<Eigen::Matrix<T, 3, 1>> eResiduals(residual);
        eResiduals = Eigen::Matrix<T, 3, 1>::Zero();

        eResiduals = d_Phi;

        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d prior_,
                                       Eigen::Matrix3d sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_Initialization_Prior_R, 3, 3>(
                new Cost_Initialization_Prior_R(prior_,
                                                std::move(sqrt_information_))));
    }

    Eigen::Vector3d prior;
    Eigen::Matrix3d sqrt_information;
};
