#include "imu_propagator.h"

namespace SensorFusion {

ImuPropagator::ImuPropagator(NoiseManager noises, double gravity_mag)
    : _noises(noises) {
    _noises.sigma_w_2 = std::pow(_noises.sigma_w, 2);
    _noises.sigma_a_2 = std::pow(_noises.sigma_a, 2);
    _noises.sigma_wb_2 = std::pow(_noises.sigma_wb, 2);
    _noises.sigma_ab_2 = std::pow(_noises.sigma_ab, 2);
    last_prop_time_offset = 0.0;
    _gravity << 0.0, 0.0, gravity_mag;
}

void ImuPropagator::predict_and_compute(std::shared_ptr<State> state,
                                        const ImuData::Ptr &data_1,
                                        const ImuData::Ptr &data_2,
                                        Eigen::Matrix<double, 15, 15> *Phi,
                                        Eigen::Matrix<double, 15, 15> *Cov) {
    double dt = data_2->timestamp - data_1->timestamp;
    Eigen::Vector3d w_hat_1 = data_1->wm - state->imu_state_->bg;
    Eigen::Vector3d a_hat_1 = data_1->am - state->imu_state_->ba;
    Eigen::Vector3d w_hat_2 = data_2->wm - state->imu_state_->bg;
    Eigen::Vector3d a_hat_2 = data_2->am - state->imu_state_->ba;

    Eigen::Matrix3d R_I_G_old = state->imu_state_->R_G_I.transpose();
    Eigen::Vector3d v_G_I_old = state->imu_state_->v_G_I;
    Eigen::Vector3d p_G_I_old = state->imu_state_->p_G_I;

    Eigen::Matrix3d R_I_G_new;
    Eigen::Vector3d v_G_I_new;
    Eigen::Vector3d p_G_I_new;
    predict_mean_discrete(R_I_G_old, v_G_I_old, p_G_I_old, dt, w_hat_1, a_hat_1,
                          w_hat_2, a_hat_2, R_I_G_new, v_G_I_new, p_G_I_new);

    state->imu_state_->R_G_I = R_I_G_new.transpose();
    state->imu_state_->v_G_I = v_G_I_new;
    state->imu_state_->p_G_I = p_G_I_new;

    if (Phi != nullptr) {
        int th_id = state->imu_state_->th_id;
        int p_id = state->imu_state_->p_id;
        int v_id = state->imu_state_->v_id;
        int bg_id = state->imu_state_->bg_id;
        int ba_id = state->imu_state_->ba_id;

        Phi->block(th_id, th_id, 3, 3) = exp_so3(-w_hat_1 * dt);
        Phi->block(th_id, bg_id, 3, 3).noalias() =
            -exp_so3(-w_hat_1 * dt) * Jr_so3(-w_hat_1 * dt) * dt;
        Phi->block(bg_id, bg_id, 3, 3).setIdentity();
        Phi->block(v_id, th_id, 3, 3).noalias() =
            -R_I_G_old.transpose() * skew_x(a_hat_1 * dt);
        Phi->block(v_id, v_id, 3, 3).setIdentity();
        Phi->block(v_id, ba_id, 3, 3) = -R_I_G_old.transpose() * dt;
        Phi->block(ba_id, ba_id, 3, 3).setIdentity();
        Phi->block(p_id, th_id, 3, 3).noalias() =
            -0.5 * R_I_G_old.transpose() * skew_x(a_hat_1 * dt * dt);
        Phi->block(p_id, v_id, 3, 3) = Eigen::Matrix3d::Identity() * dt;
        Phi->block(p_id, ba_id, 3, 3) = -0.5 * R_I_G_old.transpose() * dt * dt;
        Phi->block(p_id, p_id, 3, 3).setIdentity();

        if (Cov != nullptr) {
            Eigen::Matrix<double, 15, 12> G =
                Eigen::Matrix<double, 15, 12>::Zero();

            G.block(th_id, 0, 3, 3) =
                -exp_so3(-w_hat_1 * dt) * Jr_so3(-w_hat_1 * dt) * dt;
            G.block(v_id, 3, 3, 3) = -R_I_G_old.transpose() * dt;
            G.block(p_id, 3, 3, 3) = -0.5 * R_I_G_old.transpose() * dt * dt;
            G.block(bg_id, 6, 3, 3) = Eigen::Matrix3d::Identity();
            G.block(ba_id, 9, 3, 3) = Eigen::Matrix3d::Identity();

            Eigen::Matrix<double, 12, 12> Q =
                Eigen::Matrix<double, 12, 12>::Zero();
            Q.block(0, 0, 3, 3) =
                _noises.sigma_w_2 / dt * Eigen::Matrix3d::Identity();
            Q.block(3, 3, 3, 3) =
                _noises.sigma_a_2 / dt * Eigen::Matrix3d::Identity();
            Q.block(6, 6, 3, 3) =
                _noises.sigma_wb_2 * dt * Eigen::Matrix3d::Identity();
            Q.block(9, 9, 3, 3) =
                _noises.sigma_ab_2 * dt * Eigen::Matrix3d::Identity();

            // Compute the noise injected into the state over the interval
            Eigen::Matrix<double, 15, 15> Qd = G * Q * G.transpose();
            Qd = 0.5 * (Qd + Qd.transpose());

            *Cov = *Phi * Cov->eval() * Phi->transpose() + Qd;
        }
    }
}

void ImuPropagator::predict_mean_discrete(
    const Eigen::Matrix3d &old_R, const Eigen::Vector3d &old_v,
    const Eigen::Vector3d &old_p, const double &dt,
    const Eigen::Vector3d &w_hat1, const Eigen::Vector3d &a_hat1,
    const Eigen::Vector3d &w_hat2, const Eigen::Vector3d &a_hat2,
    Eigen::Matrix3d &new_R, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p,
    const bool &mean) {
    // If we are averaging the IMU, then do so
    Eigen::Vector3d w_hat;
    Eigen::Vector3d a_hat;
    if (mean) {
        w_hat = .5 * (w_hat1 + w_hat2);
        a_hat = .5 * (a_hat1 + a_hat2);
    } else {
        w_hat = w_hat1;
        a_hat = a_hat1;
    }

    // Pre-compute things
    new_R = exp_so3(-w_hat * dt) * old_R;

    new_v = old_v + old_R.transpose() * a_hat * dt - _gravity * dt;

    new_p = old_v + old_v * dt + 0.5 * old_R.transpose() * a_hat * dt * dt -
            0.5 * _gravity * dt * dt;
}

}  // namespace SensorFusion