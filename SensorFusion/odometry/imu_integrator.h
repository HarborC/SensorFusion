#pragma once

#include <queue>
#include <utility>

#include "sophus/so3.hpp"
#include "type.h"

namespace SensorFusion {

class IMUIntegrator {
public:
    using Ptr = std::shared_ptr<IMUIntegrator>;
    IMUIntegrator();

    /** \brief constructor of IMUIntegrator
     * \param[in] vIMU: IMU messages need to be integrated
     */
    IMUIntegrator(std::vector<ImuData::Ptr> vIMU);

    IMUIntegrator& operator=(const IMUIntegrator& imu_integrator) {
        if (this != &imu_integrator) {
            this->acc_n = imu_integrator.acc_n;
            this->gyr_n = imu_integrator.gyr_n;
            this->acc_w = imu_integrator.acc_w;
            this->gyr_w = imu_integrator.gyr_w;
            this->vimuMsg = imu_integrator.vimuMsg;
            this->dq = imu_integrator.dq;
            this->dp = imu_integrator.dp;
            this->dv = imu_integrator.dv;
            this->linearized_bg = imu_integrator.linearized_bg;
            this->linearized_ba = imu_integrator.linearized_ba;
            this->covariance = imu_integrator.covariance;
            this->jacobian = imu_integrator.jacobian;
            this->noise = imu_integrator.noise;
            this->dtime = imu_integrator.dtime;
            ;
        }
        return *this;
    }

    void Reset();

    /** \brief get delta quaternion after IMU integration
     */
    const Eigen::Quaterniond& GetDeltaQ() const;

    /** \brief get delta displacement after IMU integration
     */
    const Eigen::Vector3d& GetDeltaP() const;

    /** \brief get delta velocity after IMU integration
     */
    const Eigen::Vector3d& GetDeltaV() const;

    /** \brief get time span after IMU integration
     */
    const double& GetDeltaTime() const;

    /** \brief get linearized bias gyr
     */
    const Eigen::Vector3d& GetBiasGyr() const;

    /** \brief get linearized bias acc
     */
    const Eigen::Vector3d& GetBiasAcc() const;

    /** \brief get covariance matrix after IMU integration
     */
    const Eigen::Matrix<double, 15, 15>& GetCovariance();

    /** \brief get jacobian matrix after IMU integration
     */
    const Eigen::Matrix<double, 15, 15>& GetJacobian() const;

    /** \brief get average acceleration of IMU messages for initialization
     */
    Eigen::Vector3d GetAverageAcc();

    /** \brief push IMU message to the IMU buffer vimuMsg
     * \param[in] imu: the IMU message need to be pushed
     */
    void PushIMUMsg(const ImuData::Ptr& imu);
    void PushIMUMsg(const std::vector<ImuData::Ptr>& vimu);
    const std::vector<ImuData::Ptr>& GetIMUMsg() const;

    void meanIntegrate(const double& prevTime, const double& currTime,
                       const std::vector<ImuData::Ptr>& imu_meas,
                       std::vector<Eigen::Vector3d>* gyrs,
                       std::vector<Eigen::Vector3d>* accls,
                       std::vector<double>* dts);
    void eularIntegrate(const double& prevTime, const double& currTime,
                        const std::vector<ImuData::Ptr>& imu_meas,
                        std::vector<Eigen::Vector3d>* gyrs,
                        std::vector<Eigen::Vector3d>* accls,
                        std::vector<double>* dts);

    /** \brief only integrate gyro information of each IMU message stored in
     * vimuMsg \param[in] lastTime: the left time boundary of vimuMsg
     */
    void GyroIntegration(double lastTime, double currTime);

    /** \brief pre-integration of IMU messages stored in vimuMsg
     */
    void PreIntegration(double lastTime, double currTime,
                        const Eigen::Vector3d& bg, const Eigen::Vector3d& ba);

    /** \brief normal integration of IMU messages stored in vimuMsg
     */
    void Integration() {}

public:
    double acc_n = 0.08;
    double gyr_n = 0.004;
    double acc_w = 2.0e-4;
    double gyr_w = 2.0e-5;
    constexpr static const double lidar_m = 6e-3;
    constexpr static const double gnorm = 9.805;

    enum JacobianOrder { O_P = 0, O_R = 3, O_V = 6, O_BG = 9, O_BA = 12 };

private:
    std::vector<ImuData::Ptr> vimuMsg;
    Eigen::Quaterniond dq;
    Eigen::Vector3d dp;
    Eigen::Vector3d dv;
    Eigen::Vector3d linearized_bg;
    Eigen::Vector3d linearized_ba;
    Eigen::Matrix<double, 15, 15> covariance;
    Eigen::Matrix<double, 15, 15> jacobian;
    Eigen::Matrix<double, 12, 12> noise;
    double dtime;
};

}  // namespace SensorFusion
