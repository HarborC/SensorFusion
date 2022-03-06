#ifndef _SENSOR_FUSION_FILTER_MATH_UTILS_H_
#define _SENSOR_FUSION_FILTER_MATH_UTILS_H_

#include <Eigen/Geometry>
#include <cmath>

namespace SensorFusion {

inline void update_se3(const Eigen::Matrix<double, 6, 1>& delta_x,
                       Eigen::Matrix3d* R, Eigen::Vector3d* p) {
    // Update Rotation.
    const double delta_angle = delta_x.segment<3>(0).norm();
    Eigen::Matrix3d delta_R = Eigen::Matrix3d::Identity();
    if (std::abs(delta_angle) > 1e-12) {
        delta_R =
            Eigen::AngleAxisd(delta_angle, delta_x.segment<3>(0).normalized());
    }
    *R = R->eval() * delta_R;

    // Update position.
    *p += delta_x.segment<3>(3);
}

inline Eigen::Matrix<double, 3, 3> skew_x(
    const Eigen::Matrix<double, 3, 1>& w) {
    Eigen::Matrix<double, 3, 3> w_x;
    w_x << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
    return w_x;
}

inline Eigen::Matrix<double, 3, 3> exp_so3(
    const Eigen::Matrix<double, 3, 1>& w) {
    // get theta
    Eigen::Matrix<double, 3, 3> w_x = skew_x(w);
    double theta = w.norm();
    // Handle small angle values
    double A, B;
    if (theta < 1e-12) {
        A = 1;
        B = 0.5;
    } else {
        A = sin(theta) / theta;
        B = (1 - cos(theta)) / (theta * theta);
    }
    // compute so(3) rotation
    Eigen::Matrix<double, 3, 3> R;
    if (theta == 0) {
        R = Eigen::MatrixXd::Identity(3, 3);
    } else {
        R = Eigen::MatrixXd::Identity(3, 3) + A * w_x + B * w_x * w_x;
    }
    return R;
}

inline Eigen::Matrix<double, 3, 3> Jl_so3(Eigen::Matrix<double, 3, 1> w) {
    double theta = w.norm();
    if (theta < 1e-12) {
        return Eigen::MatrixXd::Identity(3, 3);
    } else {
        Eigen::Matrix<double, 3, 1> a = w / theta;
        Eigen::Matrix<double, 3, 3> J =
            sin(theta) / theta * Eigen::MatrixXd::Identity(3, 3) +
            (1 - sin(theta) / theta) * a * a.transpose() +
            ((1 - cos(theta)) / theta) * skew_x(a);
        return J;
    }
}

inline Eigen::Matrix<double, 3, 3> Jr_so3(Eigen::Matrix<double, 3, 1> w) {
    return Jl_so3(-w);
}

}  // namespace SensorFusion

#endif