#ifndef _SENSOR_FUSION_IMU_PROPAGATOR_H_
#define _SENSOR_FUSION_IMU_PROPAGATOR_H_

#include "../../Sensors/Imu/imu.h"
#include "../../common_header.h"
#include "../State/filter_state.h"
#include "../State/state_block.h"
#include "../Utils/math_utils.h"

namespace SensorFusion {

using namespace DatasetIO;

/**
 * @brief Struct of our imu noise parameters
 */
struct NoiseManager {
    /// Gyroscope white noise (rad/s/sqrt(hz))
    double sigma_w = 1.6968e-04;

    /// Gyroscope white noise covariance
    double sigma_w_2 = std::pow(1.6968e-04, 2);

    /// Gyroscope random walk (rad/s^2/sqrt(hz))
    double sigma_wb = 1.9393e-05;

    /// Gyroscope random walk covariance
    double sigma_wb_2 = std::pow(1.9393e-05, 2);

    /// Accelerometer white noise (m/s^2/sqrt(hz))
    double sigma_a = 2.0000e-3;

    /// Accelerometer white noise covariance
    double sigma_a_2 = std::pow(2.0000e-3, 2);

    /// Accelerometer random walk (m/s^3/sqrt(hz))
    double sigma_ab = 3.0000e-03;

    /// Accelerometer random walk covariance
    double sigma_ab_2 = std::pow(3.0000e-03, 2);
};

class ImuPropagator {
    typedef FilterState State;

public:
    typedef std::shared_ptr<ImuPropagator> Ptr;

    ImuPropagator(NoiseManager noises = NoiseManager(),
                  double gravity_mag = 9.81);

    ~ImuPropagator() {}

    void predict_and_compute(std::shared_ptr<State> state,
                             const ImuData::Ptr &data_1,
                             const ImuData::Ptr &data_2,
                             Eigen::Matrix<double, 15, 15> *Phi = nullptr,
                             Eigen::Matrix<double, 15, 15> *Cov = nullptr);

protected:
    /// Estimate for time offset at last propagation time
    double last_prop_time_offset = 0.0;
    bool have_last_prop_time_offset = false;

    void predict_mean_discrete(const Eigen::Matrix3d &old_R,
                               const Eigen::Vector3d &old_v,
                               const Eigen::Vector3d &old_p, const double &dt,
                               const Eigen::Vector3d &w_hat1,
                               const Eigen::Vector3d &a_hat1,
                               const Eigen::Vector3d &w_hat2,
                               const Eigen::Vector3d &a_hat2,
                               Eigen::Matrix3d &new_R, Eigen::Vector3d &new_v,
                               Eigen::Vector3d &new_p, const bool &mean = true);

    /// Container for the noise values
    NoiseManager _noises;

    /// Gravity vector
    Eigen::Vector3d _gravity;
};

}  // namespace SensorFusion

#endif