#include "module.h"
#include "ceres_factor/imu_initialization_factor.h"

namespace SensorFusion {

void Module::getPreIntegratedImuData(std::vector<ImuData::Ptr>& meas) {
    if (!curr_imu) {
        if (!imu_data_queue->try_pop(curr_imu)) {
            return;
        }
    }

    while (curr_imu->timeStamp <= curr_timestamp) {
        meas.push_back(curr_imu);
        if (!imu_data_queue->try_pop(curr_imu)) {
            break;
        }
    }

    if (meas.size()) {
        if (meas.front()->timeStamp - last_timestamp > 0.2 ||
            curr_timestamp - meas.back()->timeStamp > 0.2) {
            meas.clear();
            return;
        }

        if (meas.front()->timeStamp < last_timestamp) {
            std::vector<ImuData::Ptr> meas_temp;
            for (int i = 0; i < meas.size(); i++) {
                if (meas[i]->timeStamp >= last_timestamp)
                    meas_temp.push_back(meas[i]);
            }
            meas = meas_temp;
        }
    }
}

bool Module::staticInitialize() {
    const double duration_time = 2.0;
    if (window_frames.back() && window_frames.back()->pre_imu_enabled) {
        if (!accumulate_imu_meas.size() ||
            (accumulate_imu_meas.size() &&
             window_frames.back()->timeStamp >
                 accumulate_imu_meas.back()->timeStamp)) {
            auto imu_meas = window_frames.back()->imuIntegrator.GetIMUMsg();
            for (int i = 0; i < imu_meas.size(); i++) {
                accumulate_imu_meas.push_back(imu_meas[i]);
            }

            while (accumulate_imu_meas.front()->timeStamp <=
                   window_frames.back()->timeStamp - duration_time - 0.5) {
                accumulate_imu_meas.pop_front();
            }
        }
    }

    if (!accumulate_imu_meas.size()) {
        std::cout << "[staticInitialize] : no imu!" << std::endl;
        return false;
    }

    if (accumulate_imu_meas.back()->timeStamp -
            accumulate_imu_meas.front()->timeStamp <
        duration_time) {
        std::cout << "[staticInitialize] : duration_time of imu accumulation "
                     "is not enough!"
                  << std::endl;
        return false;
    }

    Eigen::Vector3d sum_g1 = Eigen::Vector3d::Zero();
    for (auto it = accumulate_imu_meas.begin(); it != accumulate_imu_meas.end();
         it++) {
        sum_g1 += (*it)->accel;
    }
    auto aver_g1 = sum_g1 / ((int)accumulate_imu_meas.size());

    double var1 = 0;
    for (auto it = accumulate_imu_meas.begin(); it != accumulate_imu_meas.end();
         it++) {
        var1 += ((*it)->accel - aver_g1).transpose() * ((*it)->accel - aver_g1);
    }
    var1 = std::sqrt(var1 / ((int)accumulate_imu_meas.size()));
    // std::cout << "var1 : " << std::fixed << std::setprecision(5) << var1 <<
    // std::endl;

    if (var1 > 0.1) {
        std::cout << "[staticInitialize] : imu is moving!" << std::endl;
        return false;
    }

    if (!tryImuAlignment(accumulate_imu_meas, gyro_bias, gravity_vector))
        return false;

    imu_initialized = InitializtionType::STATIC;
    {
        for (auto it = window_frames.begin(); it != window_frames.end(); it++) {
            auto& frame = *it;
            frame->ba = Eigen::Vector3d::Zero();
            frame->bg = gyro_bias;
            frame->V = Eigen::Vector3d::Zero();
        }
    }

    return true;
}

bool Module::dynamicInitialize() {
    if (window_frames.size() < MAX_SLIDE_WINDOW_SIZE)
        return false;

    int num_can_pre = 0;
    for (std::list<Frame::Ptr>::iterator it = window_frames.begin();;) {
        double prev_timestamp = (*it)->timeStamp;

        it++;
        if (it != window_frames.end()) {
            if ((*it) && (*it)->pre_imu_enabled) {
                (*it)->imuIntegrator.PreIntegration(
                    prev_timestamp, (*it)->timeStamp, Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero());
                num_can_pre++;
            }
        } else {
            break;
        }
    }

    if (!num_can_pre)
        return false;

    if (scale > 0) {
        Eigen::Vector3d sum_g = Eigen::Vector3d::Zero();
        for (std::list<Frame::Ptr>::iterator it = window_frames.begin();;) {
            it++;
            if (it != window_frames.end()) {
                if ((*it) && (*it)->pre_imu_enabled) {
                    sum_g += (*it)->imuIntegrator.GetDeltaV() /
                             (*it)->imuIntegrator.GetDeltaTime();
                }
            } else {
                break;
            }
        }
        auto aver_g = sum_g / num_can_pre;

        double var = 0;
        for (std::list<Frame::Ptr>::iterator it = window_frames.begin();;) {
            it++;
            if (it != window_frames.end()) {
                if ((*it) && (*it)->pre_imu_enabled) {
                    auto tmp_g = (*it)->imuIntegrator.GetDeltaV() /
                                 (*it)->imuIntegrator.GetDeltaTime();

                    var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
                }
            } else {
                break;
            }
        }
        var = std::sqrt(var / num_can_pre);

        if (var < 0.25) {
            std::cout << "[dynamicInitialize] : imu excitation not enouth!"
                      << std::endl;
            return false;
        }
    }

    {
        int push_cout = 1;
        std::list<Frame::Ptr> frame_to_imu_align;
        std::vector<int> frame_idx;
        frame_to_imu_align.push_back(
            Frame::Ptr(new Frame(window_frames.front())));
        frame_idx.push_back(0);
        for (int i = 1; i < window_frames.size();) {
            Frame::Ptr new_frame(new Frame());
            auto pre_frame = window_frames.begin();
            std::advance(pre_frame, i - 1);
            for (int j = 0; j < push_cout && i < window_frames.size(); j++) {
                auto start_frame = window_frames.begin();
                std::advance(start_frame, i);
                Frame::Ptr curr_frame = *(start_frame);

                new_frame->timeStamp = curr_frame->timeStamp;
                new_frame->imuIntegrator.PushIMUMsg(
                    curr_frame->imuIntegrator.GetIMUMsg());

                new_frame->sensor_id = curr_frame->sensor_id;
                new_frame->ExT_ = curr_frame->ExT_;
                new_frame->P_ = curr_frame->P_;
                new_frame->Q_ = curr_frame->Q_;

                i++;
            }
            new_frame->imuIntegrator.PreIntegration(
                (*pre_frame)->timeStamp, new_frame->timeStamp,
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            frame_to_imu_align.push_back(new_frame);
            frame_idx.push_back(i);
        }

        if (!tryImuAlignment(frame_to_imu_align, scale, gravity_vector))
            return false;

        for (int i = 0; i < window_frames.size(); i++) {
            auto cur_frame = window_frames.begin();
            std::advance(cur_frame, i);

            std::string id_l = (*cur_frame)->sensor_id;
            Eigen::Matrix4d exTbl = ex_pose[id_l];
            Eigen::Matrix3d exRlbi = exTbl.block<3, 3>(0, 0).transpose();
            Eigen::Vector3d exPlbi = (-exRlbi * exTbl.block<3, 1>(0, 3));

            Eigen::Vector3d Pwl = scale * (*cur_frame)->P_;
            Eigen::Quaterniond Qwl = (*cur_frame)->Q_;
            (*cur_frame)->P = Pwl + Qwl * exPlbi;
            (*cur_frame)->Q = Qwl * exRlbi;
            (*cur_frame)->bg = frame_to_imu_align.front()->bg;
            (*cur_frame)->ba = frame_to_imu_align.front()->ba;

            if (i != 0) {
                auto pre_frame = window_frames.begin();
                std::advance(pre_frame, i - 1);
                (*cur_frame)->V =
                    ((*cur_frame)->P - (*pre_frame)->P) /
                    ((*cur_frame)->timeStamp - (*pre_frame)->timeStamp);
            }
        }

        for (int i = 0; i < frame_idx.size(); i++) {
            auto cur_frame = window_frames.begin();
            std::advance(cur_frame, frame_idx[i]);

            auto cur_frame2 = frame_to_imu_align.begin();
            std::advance(cur_frame2, i);

            (*cur_frame)->V = (*cur_frame2)->V;
        }
    }

    imu_initialized = InitializtionType::DYNAMIC;

    return true;
}

bool tryImuAlignment(std::list<Frame::Ptr>& frames, double& scale,
                     Eigen::Vector3d& GravityVector) {
    int sum_acc_num = 0;
    Eigen::Vector3d average_acc(0, 0, 0);
    for (auto it = frames.begin(); it != frames.end(); it++) {
        for (auto& imu : (*it)->imuIntegrator.GetIMUMsg()) {
            Eigen::Vector3d acc = imu->accel;
            average_acc += acc;
            sum_acc_num++;
            if (sum_acc_num > 30)
                break;
        }
    }

    if (sum_acc_num <= 30)
        return false;
    average_acc = average_acc / sum_acc_num;
    double info_g = std::fabs(9.805 - average_acc.norm());
    average_acc = average_acc * 9.805 / average_acc.norm();

    // calculate the initial gravitydirection
    double para_quat[4];
    para_quat[0] = 1;
    para_quat[1] = 0;
    para_quat[2] = 0;
    para_quat[3] = 0;

    ceres::LocalParameterization* quatParam =
        new ceres::QuaternionParameterization();
    ceres::Problem problem_quat;

    problem_quat.AddParameterBlock(para_quat, 4, quatParam);

    problem_quat.AddResidualBlock(Cost_Initial_G::Create(average_acc), nullptr,
                                  para_quat);

    ceres::Solver::Options options_quat;
    ceres::Solver::Summary summary_quat;
    ceres::Solve(options_quat, &problem_quat, &summary_quat);

    Eigen::Quaterniond q_wg(para_quat[0], para_quat[1], para_quat[2],
                            para_quat[3]);

    // build prior factor of LIO initialization
    Eigen::Vector3d prior_r = Eigen::Vector3d::Zero();
    Eigen::Vector3d prior_ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d prior_bg = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> prior_v;
    int v_size = frames.size();
    for (int i = 0; i < v_size; i++) {
        prior_v.push_back(Eigen::Vector3d::Zero());
    }
    Sophus::SO3d SO3_R_wg(q_wg.toRotationMatrix());
    prior_r = SO3_R_wg.log();

    for (int i = 1; i < v_size; i++) {
        auto iter = frames.begin();
        auto iter_next = frames.begin();
        std::advance(iter, i - 1);
        std::advance(iter_next, i);

        Eigen::Vector3d exPlbj =
            (-(*iter_next)->ExT_.block<3, 3>(0, 0).transpose() *
             (*iter_next)->ExT_.block<3, 1>(0, 3));

        Eigen::Vector3d exPlbi = (-(*iter)->ExT_.block<3, 3>(0, 0).transpose() *
                                  (*iter)->ExT_.block<3, 1>(0, 3));

        Eigen::Vector3d velo_imu =
            ((*iter_next)->P_ - (*iter)->P_ + (*iter_next)->Q_ * exPlbj -
             (*iter)->Q_ * exPlbi) /
            ((*iter_next)->timeStamp - (*iter)->timeStamp);
        prior_v[i] = velo_imu;
    }
    prior_v[0] = prior_v[1];

    double para_pose[v_size][6];
    double para_ex[6];
    double para_v[v_size][3];
    double para_r[3];
    double para_ba[3];
    double para_bg[3];
    double para_scale[1];

    Eigen::Map<Eigen::Matrix<double, 6, 1>> ex_temp(para_ex);
    ex_temp.segment<3>(0) = (*frames.begin())->ExT_.block<3, 1>(0, 3);
    ex_temp.segment<3>(3) =
        Sophus::SO3d(
            Eigen::Quaterniond((*frames.begin())->ExT_.block<3, 3>(0, 0))
                .normalized())
            .log();

    for (int i = 0; i < v_size; i++) {
        auto iter = frames.begin();
        std::advance(iter, i);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> PR(para_pose[i]);
        PR.segment<3>(0) = (*iter)->P_;
        PR.segment<3>(3) = Sophus::SO3d((*iter)->Q_).log();
    }

    for (int i = 0; i < 3; i++) {
        para_r[i] = 0;
        para_ba[i] = 0;
        para_bg[i] = 0;
    }

    for (int i = 0; i < v_size; i++) {
        for (int j = 0; j < 3; j++) {
            para_v[i][j] = prior_v[i][j];
        }
    }

    para_scale[0] = ((scale <= 0) ? 1.0 : scale);

    Eigen::Matrix<double, 3, 3> sqrt_information_r =
        2000.0 * Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 3> sqrt_information_ba =
        1000.0 * Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 3> sqrt_information_bg =
        4000.0 * Eigen::Matrix<double, 3, 3>::Identity();
    // Eigen::Matrix<double, 3, 3> sqrt_information_v =
    //     4000.0 * Eigen::Matrix<double, 3, 3>::Identity();

    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(para_r, 3);
    problem.AddParameterBlock(para_ba, 3);
    problem.AddParameterBlock(para_bg, 3);
    problem.AddParameterBlock(para_ex, 6);
    problem.SetParameterBlockConstant(para_ex);
    for (int i = 0; i < v_size; i++) {
        problem.AddParameterBlock(para_pose[i], 6);
        problem.SetParameterBlockConstant(para_pose[i]);
        problem.AddParameterBlock(para_v[i], 3);
    }

    problem.AddParameterBlock(para_scale, 1);
    if (scale <= 0)
        problem.SetParameterBlockConstant(para_scale);

    problem.AddResidualBlock(
        Cost_Initialization_Prior_R::Create(prior_r, sqrt_information_r),
        nullptr, para_r);

    problem.AddResidualBlock(
        Cost_Initialization_Prior_bv::Create(prior_ba, sqrt_information_ba),
        nullptr, para_ba);
    problem.AddResidualBlock(
        Cost_Initialization_Prior_bv::Create(prior_bg, sqrt_information_bg),
        nullptr, para_bg);

    // for (int i = 0; i < v_size; i++) {
    //     problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(
    //                                  prior_v[i], sqrt_information_v),
    //                              nullptr, para_v[i]);
    // }

    for (int i = 1; i < v_size; i++) {
        auto iter_next = frames.begin();
        std::advance(iter_next, i);

        problem.AddResidualBlock(Cost_Initialization_IMU::Create(
                                     (*iter_next)->imuIntegrator,
                                     Eigen::LLT<Eigen::Matrix<double, 9, 9>>(
                                         (*iter_next)
                                             ->imuIntegrator.GetCovariance()
                                             .block<9, 9>(0, 0)
                                             .inverse())
                                         .matrixL()
                                         .transpose()),
                                 nullptr, para_pose[i - 1], para_pose[i],
                                 para_ex, para_v[i - 1], para_v[i], para_ba,
                                 para_bg, para_r, para_scale);
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 6;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector3d r_wg(para_r[0], para_r[1], para_r[2]);
    GravityVector = Sophus::SO3d::exp(r_wg) * Eigen::Vector3d(0, 0, -9.805);

    Eigen::Vector3d ba_vec(para_ba[0], para_ba[1], para_ba[2]);
    Eigen::Vector3d bg_vec(para_bg[0], para_bg[1], para_bg[2]);

    if (ba_vec.norm() > 0.5 || bg_vec.norm() > 0.5) {
        std::cout << "Too Large Biases! Initialization Failed!" << std::endl;
        return false;
    }

    if (para_scale[0] < 0.01 || para_scale[0] > 100) {
        std::cout << "Too Large Scale! Initialization Failed!" << std::endl;
        return false;
    }

    scale = para_scale[0];

    for (int i = 0; i < v_size; i++) {
        auto iter = frames.begin();
        std::advance(iter, i);
        (*iter)->ba = ba_vec;
        (*iter)->bg = bg_vec;
        Eigen::Vector3d bv_vec(para_v[i][0], para_v[i][1], para_v[i][2]);
        if ((bv_vec - prior_v[i]).norm() > 2.0) {
            std::cout << "Too Large Velocity! Initialization Failed!"
                      << std::endl;
            std::cout << "delta v norm: " << (bv_vec - prior_v[i]).norm()
                      << std::endl;
            return false;
        }
        (*iter)->V = scale * bv_vec;
    }

    std::cout << "\n=============================\n| Dynamic Initialization "
                 "Successful | "
              << "\n == == == == == == == == == == == == == == =\n "
              << std::endl;

    return true;
}

bool tryImuAlignment(const std::list<ImuData::Ptr>& imu_meas,
                     Eigen::Vector3d& gyroBias,
                     Eigen::Vector3d& GravityVector) {
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d lin_accel = Eigen::Vector3d::Zero();
    for (auto it = imu_meas.begin(); it != imu_meas.end(); it++) {
        gyro_bias[0] += (*it)->gyro.x();
        gyro_bias[1] += (*it)->gyro.y();
        gyro_bias[2] += (*it)->gyro.z();
        lin_accel[0] += (*it)->accel.x();
        lin_accel[1] += (*it)->accel.y();
        lin_accel[2] += (*it)->accel.z();
    }
    gyroBias = gyro_bias / (double)imu_meas.size();

    Eigen::Vector3d average_acc = lin_accel / (double)imu_meas.size();
    double info_g = std::fabs(9.805 - average_acc.norm());
    average_acc = average_acc * 9.805 / average_acc.norm();

    // calculate the initial gravitydirection
    double para_quat[4];
    para_quat[0] = 1;
    para_quat[1] = 0;
    para_quat[2] = 0;
    para_quat[3] = 0;

    ceres::LocalParameterization* quatParam =
        new ceres::QuaternionParameterization();
    ceres::Problem problem_quat;

    problem_quat.AddParameterBlock(para_quat, 4, quatParam);

    problem_quat.AddResidualBlock(Cost_Initial_G::Create(average_acc), nullptr,
                                  para_quat);

    ceres::Solver::Options options_quat;
    ceres::Solver::Summary summary_quat;
    ceres::Solve(options_quat, &problem_quat, &summary_quat);

    Eigen::Quaterniond q_wg(para_quat[0], para_quat[1], para_quat[2],
                            para_quat[3]);
    GravityVector = q_wg.toRotationMatrix() * Eigen::Vector3d(0, 0, -9.805);

    std::cout << "\n=============================\n| Static Initialization "
                 "Successful | "
              << "\n == == == == == == == == == == == == == == =\n "
              << std::endl;

    return true;
}

}  // namespace SensorFusion