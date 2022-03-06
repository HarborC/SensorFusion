#ifndef _SENSOR_FUSION_FILTER_STATE_BLOCK_H_
#define _SENSOR_FUSION_FILTER_STATE_BLOCK_H_

#include <memory>
#include <unordered_map>

#include "../../common_header.h"
#include "../Utils/math_utils.h"

namespace SensorFusion {

struct StateBlock {
    // Global ID.
    long int id = -1;
    // Error state size.
    int size = 0;
    // Index in state vector/covariance.
    int state_idx = 0;
};

struct ImuState : public StateBlock {
    typedef std::shared_ptr<ImuState> Ptr;

    ImuState() { size = 6; }

    Eigen::Matrix3d R_G_I;
    Eigen::Vector3d p_G_I;
    Eigen::Vector3d v_G_I;
    Eigen::Vector3d ba;
    Eigen::Vector3d bg;

    int th_id = 0;
    int p_id = 3;
    int v_id = 6;
    int bg_id = 9;
    int ba_id = 12;

    void Update(const Eigen::Matrix<double, 6, 1>& delta_x) {
        update_se3(delta_x, &R_G_I, &p_G_I);
    }
};

struct WheelState : public StateBlock {
    typedef std::shared_ptr<WheelState> Ptr;
    WheelState() { size = 6; }

    Eigen::Matrix3d G_R_O;
    Eigen::Vector3d G_p_O;

    void Update(const Eigen::Matrix<double, 6, 1>& delta_x) {
        update_se3(delta_x, &G_R_O, &G_p_O);
    }
};

struct CameraState : public StateBlock {
    typedef std::shared_ptr<CameraState> Ptr;

    CameraState() { size = 6; }

    double timestamp;

    // SE3 of this frame.
    Eigen::Matrix3d R_G_C;
    Eigen::Vector3d p_G_C;

    // Features in this frame. ID to point in image.
    std::unordered_map<long int, Eigen::Vector2d> id_pt_map;

    void Update(const Eigen::Matrix<double, 6, 1>& delta_x) {
        update_se3(delta_x, &R_G_C, &p_G_C);
    }
};

struct Extrinsic : public StateBlock {
    typedef std::shared_ptr<Extrinsic> Ptr;
    Extrinsic() { size = 6; }

    Eigen::Matrix3d R_B_S;
    Eigen::Vector3d p_B_S;

    void Update(const Eigen::Matrix<double, 6, 1>& delta_x) {
        update_se3(delta_x, &R_B_S, &p_B_S);
    }
};

struct WheelIntrinsic : public StateBlock {
    WheelIntrinsic() { size = 3; }

    double kl;
    double kr;
    double b;

    void Update(const Eigen::Vector3d& delta_x) {
        kl += delta_x[0];
        kr += delta_x[1];
        b += delta_x[2];
    }
};

}  // namespace SensorFusion

#endif
