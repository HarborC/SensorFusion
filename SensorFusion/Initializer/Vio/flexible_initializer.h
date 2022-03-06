//
// Created by xiaochen at 19-8-13.
// A flexible initializer that can automatically initialize in case of static or
// dynamic scene.
//

#ifndef FLEXIBLE_INITIALIZER_H
#define FLEXIBLE_INITIALIZER_H

#include <iostream>

#include "dynamic_initializer.h"
#include "imu_data.h"
#include "static_initializer.h"

using namespace std;

namespace larvio {

class FlexibleInitializer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    FlexibleInitializer() = delete;
    FlexibleInitializer(const double& max_feature_dis_, const int& static_Num_,
                        const double& td_, const Eigen::Matrix3d& Ma_,
                        const Eigen::Matrix3d& Tg_, const Eigen::Matrix3d& As_,
                        const double& acc_n_, const double& acc_w_,
                        const double& gyr_n_, const double& gyr_w_,
                        const Eigen::Matrix3d& R_c2b,
                        const Eigen::Vector3d& t_bc_b,
                        const double& imu_img_timeTh_) {
        staticInitPtr.reset(new StaticInitializer(max_feature_dis_, static_Num_,
                                                  td_, Ma_, Tg_, As_));

        dynamicInitPtr.reset(
            new DynamicInitializer(td_, Ma_, Tg_, As_, acc_n_, acc_w_, gyr_n_,
                                   gyr_w_, R_c2b, t_bc_b, imu_img_timeTh_));
    }

    // Destructor
    ~FlexibleInitializer() {}

    // Interface for trying to initialize
    bool tryIncInit(std::vector<ImuData>& imu_msg_buffer,
                    MonoCameraMeasurementPtr img_msg,
                    Eigen::Vector3d& m_gyro_old, Eigen::Vector3d& m_acc_old,
                    IMUState& imu_state);

private:
    // Inclinometer-initializer
    std::shared_ptr<StaticInitializer> staticInitPtr;
    // Dynamic initializer
    std::shared_ptr<DynamicInitializer> dynamicInitPtr;
};

}  // namespace larvio

#endif  // FLEXIBLE_INITIALIZER_H
