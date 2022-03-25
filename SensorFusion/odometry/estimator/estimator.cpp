#include "estimator.h"
#include "../global.h"
#include "../map_viewer.h"
#include "../pointcloud_viewer.h"
#include "initial_sfm.h"

namespace SensorFusion {

bool show_opt = true;
// PointCloudViewer<pcl::PointXYZRGB> viewer;
MapViewer map_viewer;

Estimator::Estimator(const int Ex_Mode) {}

void Estimator::Estimate() {
    // V1: 假设Camera、Lidar、IMU已经完成时间同步和标定工作
    if (!dataSynchronize()) {
        return;
    }

    imuInitialize();

    preProcess();
    optimization();
    postProcess();

    // 去除粗差

    // 失败检测
    // if (failureDetection()) {}

    // 滑窗取出老的或者新的帧
    slideWindow();

    {
        std::vector<Visualizer::Pose> lidar_poses;
        for (int m = 0; m < modules.size(); m++) {
            if (modules[m]->sensorType == SensorFlag::LIDAR) {
                for (int i = 0; i < modules[m]->slide_frames.size(); i++) {
                    auto f = modules[m]->slide_frames.begin();
                    std::advance(f, i);
                    Visualizer::Pose p((*f)->timeStamp,
                                       (*f)->Q.toRotationMatrix(), (*f)->P);
                    lidar_poses.push_back(p);
                }

                for (int i = 0; i < modules[m]->window_frames.size(); i++) {
                    auto f = modules[m]->window_frames.begin();
                    std::advance(f, i);
                    Visualizer::Pose p((*f)->timeStamp,
                                       (*f)->Q.toRotationMatrix(), (*f)->P);
                    lidar_poses.push_back(p);
                }
            }
        }
        g_viz->DrawCameras(lidar_poses);
    }
}

void Estimator::slideWindow() {
    for (size_t i = 0; i < modules.size(); i++) {
        if (modules[i]->prime_flag) {
            modules[i]->slideWindow();
            auto& prime_module = modules[i];
            for (size_t i = 0; i < modules.size(); i++) {
                if (!modules[i]->prime_flag)
                    modules[i]->slideWindow(prime_module);
            }
        }
    }
}

bool Estimator::imuInitialize() {
    // return false;
    if (imu_initialized)
        return true;

    for (size_t i = 0; i < modules.size(); i++) {
        imu_initialized = modules[i]->initialize();
        if (imu_initialized) {
            initializeOtherModule(modules[i]);
            return true;
        }
    }

    return false;
}

bool Estimator::dataSynchronize() {
    double timestamp = 0;
    for (size_t i = 0; i < modules.size(); i++) {
        if (modules[i]->prime_flag) {
            timestamp = modules[i]->dataSynchronize();
            break;
        }
    }

    if (timestamp > 0) {
        for (size_t i = 0; i < modules.size(); i++) {
            if (!modules[i]->prime_flag) {
                double timestamp_sensor =
                    modules[i]->dataSynchronize(timestamp);
            }
        }
        return true;
    }

    return false;
}

void Estimator::preProcess() {
    if (!imu_initialized)
        return;

    for (size_t i = 0; i < modules.size(); i++) {
        modules[i]->preProcess();
    }
}

void Estimator::postProcess() {
    if (!imu_initialized)
        return;

    for (size_t i = 0; i < modules.size(); i++) {
        modules[i]->postProcess();
    }
}

void Estimator::optimization() {
    if (!imu_initialized)
        return;

    mergeWindow();

    for (int iterOpt = 0; iterOpt < max_iters; ++iterOpt) {
        // 构建problem
        problem.reset(new ceres::Problem);
        for (size_t i = 0; i < modules.size(); i++) {
            modules[i]->problem = problem;
        }

        // vector转换为double
        for (size_t i = 0; i < modules.size(); i++) {
            modules[i]->vector2double();
            modules[i]->addParameter();
        }

        problem->GetParameterBlocks(&(evaluate_options.parameter_blocks));

        // 添加residuals
        for (size_t i = 0; i < modules.size(); i++) {
            modules[i]->addResidualBlock(iterOpt);
        }
        addImuResidualBlock();
        addMargResidualBlock();

        // Before
        if (show_opt) {
            std::cout << "Opt Before" << std::endl;
            // imu
            evaluate_options.residual_blocks = imu_residual_block_ids;
            double imu_cost;
            std::vector<double> imu_residuals;
            problem->Evaluate(evaluate_options, &imu_cost, &imu_residuals, NULL,
                              NULL);

            std::cout << "imu_pre"
                      << " : " << std::fixed << std::setprecision(6) << imu_cost
                      << " | ";
            for (int i = 0; i < imu_residuals.size(); i++)
                std::cout << std::fixed << std::setprecision(6)
                          << imu_residuals[i] << " ";
            std::cout << std::endl;

            // module
            for (size_t i = 0; i < modules.size(); i++) {
                for (auto t : modules[i]->residual_block_ids) {
                    if (t.second.size()) {
                        evaluate_options.residual_blocks = t.second;
                        double cost;
                        std::vector<double> residuals;
                        problem->Evaluate(evaluate_options, &cost, &residuals,
                                          NULL, NULL);

                        std::cout << t.first << " : " << std::fixed
                                  << std::setprecision(6) << cost << " | ";
                        // for (int j = 0; j < residuals.size(); j++)
                        //     std::cout << std::fixed << std::setprecision(6)
                        //               << residuals[j] << " ";
                        std::cout << std::endl;
                    }
                }
            }

            double total_cost;
            evaluate_options.residual_blocks.clear();
            problem->Evaluate(evaluate_options, &total_cost, NULL, NULL, NULL);
            std::cout << "total_cost"
                      << " : " << std::fixed << std::setprecision(6)
                      << total_cost << " | ";
            std::cout << std::endl;
        }

        TicToc t0;

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        // options.minimizer_progress_to_stdout = false;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = max_num_iterations;
        options.num_threads = max_num_threads;
        ceres::Solver::Summary summary;
        ceres::Solve(options, problem.get(), &summary);

        auto dt = t0.toc();
        std::cout << "opt time : " << dt << std::endl;

        // After
        if (show_opt) {
            std::cout << "Opt After" << std::endl;
            // imu
            evaluate_options.residual_blocks = imu_residual_block_ids;
            double imu_cost;
            std::vector<double> imu_residuals;
            problem->Evaluate(evaluate_options, &imu_cost, &imu_residuals, NULL,
                              NULL);

            std::cout << "imu_pre"
                      << " : " << std::fixed << std::setprecision(6) << imu_cost
                      << " | ";
            for (int i = 0; i < imu_residuals.size(); i++)
                std::cout << std::fixed << std::setprecision(6)
                          << imu_residuals[i] << " ";
            std::cout << std::endl;

            // module
            for (size_t i = 0; i < modules.size(); i++) {
                for (auto t : modules[i]->residual_block_ids) {
                    if (t.second.size()) {
                        evaluate_options.residual_blocks = t.second;
                        double cost;
                        std::vector<double> residuals;
                        problem->Evaluate(evaluate_options, &cost, &residuals,
                                          NULL, NULL);

                        std::cout << t.first << " : " << std::fixed
                                  << std::setprecision(6) << cost << " | ";
                        // for (int j = 0; j < residuals.size(); j++)
                        //     std::cout << std::fixed << std::setprecision(6)
                        //               << residuals[j] << " ";
                        std::cout << std::endl;
                    }
                }
            }

            double total_cost;
            evaluate_options.residual_blocks.clear();
            problem->Evaluate(evaluate_options, &total_cost, NULL, NULL, NULL);
            std::cout << "total_cost"
                      << " : " << std::fixed << std::setprecision(6)
                      << total_cost << " | ";
            std::cout << std::endl;
        }

        for (size_t i = 0; i < modules.size(); i++) {
            modules[i]->double2vector();
        }

        bool fine_solve_flag = false;
        for (size_t i = 0; i < modules.size(); i++) {
            if (modules[i]->getFineSolveFlag()) {
                fine_solve_flag = true;
                break;
            }
        }

        if (fine_solve_flag || (iterOpt + 1) == max_iters) {
            marginalization();
            break;
        }
    }
}

void Estimator::mergeWindow() {
    all_frames.clear();
    all_imu_preintegrators.clear();

    for (size_t i = 0; i < modules.size(); i++) {
        for (size_t j = 0; j < modules[i]->window_frames.size(); j++) {
            auto curr_frame = modules[i]->window_frames.begin();
            std::advance(curr_frame, j);
            all_frames.push_back(
                IntFramePtr(std::pair<int, int>(i, j), *curr_frame));
        }
    }

    std::sort(all_frames.begin(), all_frames.end(),
              [](IntFramePtr a, IntFramePtr b) -> bool {
                  return a.second->timeStamp < b.second->timeStamp;
              });

    std::vector<ImuData::Ptr> all_imu_datas_temp;
    for (size_t i = 0; i < all_frames.size(); i++) {
        const auto& imu_datas = all_frames[i].second->imuIntegrator.GetIMUMsg();
        for (size_t j = 0; j < imu_datas.size(); j++) {
            all_imu_datas_temp.push_back(imu_datas[j]);
        }
    }

    std::sort(all_imu_datas_temp.begin(), all_imu_datas_temp.end(),
              [](ImuData::Ptr a, ImuData::Ptr b) -> bool {
                  return a->timeStamp < b->timeStamp;
              });

    std::vector<ImuData::Ptr> all_imu_datas;
    for (size_t i = 0; i < all_imu_datas_temp.size() - 1; i++) {
        if (i == 0) {
            all_imu_datas.push_back(all_imu_datas_temp[i]);
        }

        if (all_imu_datas_temp[i + 1] != all_imu_datas_temp[i]) {
            all_imu_datas.push_back(all_imu_datas_temp[i + 1]);
        }
    }

    int imu_index = 0;
    all_imu_preintegrators.push_back(IMUIntegrator());
    for (int i = 1; i < all_frames.size(); i++) {
        std::vector<ImuData::Ptr> imu_meas;
        for (int j = imu_index; j < all_imu_datas.size(); j++) {
            if (all_imu_datas[j]->timeStamp >
                    all_frames[i - 1].second->timeStamp &&
                all_imu_datas[j]->timeStamp <=
                    all_frames[i].second->timeStamp) {
                imu_meas.push_back(all_imu_datas[j]);
            } else if (all_imu_datas[j]->timeStamp >
                       all_frames[i].second->timeStamp) {
                break;
            }
            imu_index++;
        }

        IMUIntegrator imu_preintegrator;
        if (imu_meas.size()) {
            imu_preintegrator.PushIMUMsg(imu_meas);
            imu_preintegrator.PreIntegration(
                all_frames[i - 1].second->timeStamp,
                all_frames[i].second->timeStamp, all_frames[i - 1].second->bg,
                all_frames[i - 1].second->ba);
        }
        all_imu_preintegrators.push_back(imu_preintegrator);
    }
}

void Estimator::addImuResidualBlock() {
    imu_residual_block_ids.clear();

    for (size_t i = 1; i < all_frames.size(); i++) {
        int module_id0 = all_frames[i - 1].first.first;
        int module_id1 = all_frames[i].first.first;
        int win_id0 = all_frames[i - 1].first.second;
        int win_id1 = all_frames[i].first.second;
        auto re_id = problem->AddResidualBlock(
            Cost_NavState_PRV_Bias::Create(
                all_imu_preintegrators[i],
                const_cast<Eigen::Vector3d&>(gravity_vector),
                Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
                    all_imu_preintegrators[i].GetCovariance().inverse())
                    .matrixL()
                    .transpose()),
            nullptr, modules[module_id0]->para_Pose[win_id0],
            modules[module_id0]->para_SpeedBias[win_id0],
            modules[module_id1]->para_Pose[win_id1],
            modules[module_id1]->para_SpeedBias[win_id1]);
        imu_residual_block_ids.push_back(re_id);
    }
}

void Estimator::addMargResidualBlock() {
    // construct new marginlization_factor
    if (last_marginalization_info) {
        auto* marginalization_factor =
            new MarginalizationFactor(last_marginalization_info);
        problem->AddResidualBlock(marginalization_factor, nullptr,
                                  last_marginalization_parameter_blocks);
    }
}

void Estimator::marginalization() {
    double first_timestamp = -1;
    bool to_be_marg = true;
    for (size_t i = 0; i < modules.size(); i++) {
        if (modules[i]->prime_flag) {
            if (modules[i]->window_frames.size() < SLIDE_WINDOW_SIZE) {
                to_be_marg = false;
            } else {
                auto curr_frame = modules[i]->window_frames.begin();
                std::advance(curr_frame, 1);
                first_timestamp = (*curr_frame)->timeStamp;
            }

            break;
        }
    }

    if (to_be_marg) {
        for (size_t i = 0; i < modules.size(); i++) {
            if (modules[i]->prime_flag) {
                modules[i]->slide_window_size = 1;
            } else {
                modules[i]->slide_window_size =
                    modules[i]->getMargWindowSize(first_timestamp);
            }
        }

        auto* marginalization_info = new MarginalizationInfo();

        // vector2double();

        if (last_marginalization_info) {
            std::vector<int> drop_set;
            for (int i = 0;
                 i <
                 static_cast<int>(last_marginalization_parameter_blocks.size());
                 i++) {
                for (int m = 0; m < modules.size(); m++) {
                    for (int j = 0; j < modules[m]->slide_window_size; j++) {
                        if (last_marginalization_parameter_blocks[i] ==
                                modules[m]->para_Pose[j] ||
                            last_marginalization_parameter_blocks[i] ==
                                modules[m]->para_SpeedBias[j])
                            drop_set.push_back(i);
                    }
                }
            }

            // construct new marginlization_factor
            MarginalizationFactor* marginalization_factor =
                new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
                marginalization_factor, NULL,
                last_marginalization_parameter_blocks, drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        for (size_t i = 0; i < modules.size(); i++) {
            modules[i]->marginalization1(last_marginalization_info,
                                         last_marginalization_parameter_blocks,
                                         marginalization_info,
                                         modules[i]->slide_window_size);
        }

        // imu marg
        {
            for (size_t i = 1; i < all_frames.size(); i++) {
                int module_id0 = all_frames[i].first.first;
                int module_id1 = all_frames[i + 1].first.first;
                int win_id0 = all_frames[i].first.second;
                int win_id1 = all_frames[i + 1].first.second;
                if (all_frames[i].second->timeStamp >= first_timestamp) {
                    ceres::CostFunction* IMU_Cost =
                        Cost_NavState_PRV_Bias::Create(
                            all_imu_preintegrators[i],
                            const_cast<Eigen::Vector3d&>(gravity_vector),
                            Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
                                all_imu_preintegrators[i]
                                    .GetCovariance()
                                    .inverse())
                                .matrixL()
                                .transpose());
                    auto* residual_block_info = new ResidualBlockInfo(
                        IMU_Cost, nullptr,
                        std::vector<double*>{
                            modules[module_id0]->para_Pose[win_id0],
                            modules[module_id0]->para_SpeedBias[win_id0],
                            modules[module_id1]->para_Pose[win_id1],
                            modules[module_id1]->para_SpeedBias[win_id1]},
                        std::vector<int>{0, 1});
                    marginalization_info->addResidualBlockInfo(
                        residual_block_info);
                    break;
                } else {
                    ceres::CostFunction* IMU_Cost =
                        Cost_NavState_PRV_Bias::Create(
                            all_imu_preintegrators[i],
                            const_cast<Eigen::Vector3d&>(gravity_vector),
                            Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
                                all_imu_preintegrators[i]
                                    .GetCovariance()
                                    .inverse())
                                .matrixL()
                                .transpose());
                    auto* residual_block_info = new ResidualBlockInfo(
                        IMU_Cost, nullptr,
                        std::vector<double*>{
                            modules[module_id0]->para_Pose[win_id0],
                            modules[module_id0]->para_SpeedBias[win_id0],
                            modules[module_id1]->para_Pose[win_id1],
                            modules[module_id1]->para_SpeedBias[win_id1]},
                        std::vector<int>{0, 1, 2, 3});
                    marginalization_info->addResidualBlockInfo(
                        residual_block_info);
                }
            }
        }

        marginalization_info->preMarginalize();
        marginalization_info->marginalize();

        std::unordered_map<long, double*> addr_shift;
        for (size_t i = 0; i < modules.size(); i++) {
            modules[i]->marginalization2(addr_shift,
                                         modules[i]->slide_window_size);
        }

        std::vector<double*> parameter_blocks =
            marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
}

void Estimator::initializeOtherModule(const Module::Ptr& init_module) {
    gravity_vector = init_module->gravity_vector;

    for (int i = 0; i < modules.size(); i++) {
        if (modules[i] != init_module) {
            if (init_module->imu_initialized ==
                Module::InitializtionType::STATIC) {
                modules[i]->imu_initialized = init_module->imu_initialized;
                modules[i]->gravity_vector = init_module->gravity_vector;

                for (auto it = modules[i]->window_frames.begin();
                     it != modules[i]->window_frames.end(); it++) {
                    auto& frame = *it;
                    frame->ba = Eigen::Vector3d::Zero();
                    frame->bg = init_module->gyro_bias;
                    frame->V = Eigen::Vector3d::Zero();
                }
            } else {
                std::cout << "dsdsd" << std::endl;
                exit(0);
            }
        }
    }

    for (size_t i = 0; i < modules.size(); i++) {
        if (modules[i]->prime_flag) {
            while (modules[i]->frame_count > SLIDE_WINDOW_SIZE) {
                auto slide_f = modules[i]->window_frames.front();
                modules[i]->slide_frames.push_back(slide_f);
                modules[i]->window_frames.pop_front();
                modules[i]->frame_count--;
            }
            auto& prime_module = modules[i];
            for (size_t i = 0; i < modules.size(); i++) {
                if (!modules[i]->prime_flag)
                    modules[i]->slideWindow(prime_module);
            }
        }
    }
}

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

int CameraModule::initialize() {
    return 0;

    // check imu observibility
    {
        if (window_frames.size() >= 2) {
            Eigen::Vector3d sum_g = Eigen::Vector3d::Zero();
            auto it = window_frames.begin();
            for (it = window_frames.begin(), it++; it != window_frames.end();
                 ++it) {
                sum_g += (*it)->imuIntegrator.GetDeltaV() /
                         (*it)->imuIntegrator.GetDeltaTime();
            }

            auto aver_g = sum_g / ((int)window_frames.size() - 1);

            double var = 0;
            for (it = window_frames.begin(), it++; it != window_frames.end();
                 ++it) {
                auto tmp_g = (*it)->imuIntegrator.GetDeltaV() /
                             (*it)->imuIntegrator.GetDeltaTime();
                var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            }
            var = std::sqrt(var / ((int)window_frames.size() - 1));

            if (var < 0.25) {
                std::cout << "imu excitation not enouth!" << std::endl;
                // return false;
            }
        } else {
            return 0;
        }
    }

    // global sfm
    {
        std::unordered_map<std::string, std::vector<SFMFeature>> sfm_feat;

        for (auto& it_per_id : f_manager.feature) {
            int imu_j = it_per_id.start_frame - 1;
            SFMFeature tmp_feature;
            tmp_feature.state = false;
            tmp_feature.id = it_per_id.feature_id;
            auto cam_id = it_per_id.feature_per_frame[0].camids[0];
            for (auto& it_per_frame : it_per_id.feature_per_frame) {
                imu_j++;
                Eigen::Vector3d pts_j = it_per_frame.points[0];
                tmp_feature.observation.push_back(std::make_pair(
                    imu_j, Eigen::Vector2d{pts_j.x() / pts_j.z(),
                                           pts_j.y() / pts_j.z()}));
            }
            if (sfm_feat.find(cam_id) == sfm_feat.end())
                sfm_feat[cam_id] = std::vector<SFMFeature>();
            sfm_feat[cam_id].push_back(tmp_feature);
        }

        for (auto each_sfm_f : sfm_feat) {
            const auto& sensor_id = each_sfm_f.first;
            std::vector<SFMFeature>& sfm_f = each_sfm_f.second;

            std::vector<Eigen::Quaterniond> Q((int)window_frames.size(),
                                              Eigen::Quaterniond::Identity());
            std::vector<Eigen::Vector3d> T((int)window_frames.size(),
                                           Eigen::Vector3d::Zero());
            std::map<int, Eigen::Vector3d> sfm_tracked_points;

            Eigen::Matrix3d relative_R;
            Eigen::Vector3d relative_T;
            int l;
            if (!relativePose(sensor_id, relative_R, relative_T, l)) {
                std::cout
                    << "no enough features or parallax; Move device around"
                    << std::endl;
                continue;
            }

            GlobalSFM sfm;
            if (!sfm.construct(Q, T, l, relative_R, relative_T, sfm_f,
                               sfm_tracked_points)) {
                std::cout << "global SFM failed!" << std::endl;
                // marginalization_flag = MARGIN_OLD;
                continue;
            }

            {
                int push_cout = 1;
                std::list<Frame::Ptr> frame_to_imu_align;
                frame_to_imu_align.push_back(
                    Frame::Ptr(new Frame(window_frames.front())));
                frame_to_imu_align.front()->P_ = frame_to_imu_align.front()->P;
                frame_to_imu_align.front()->Q_ = frame_to_imu_align.front()->Q;
                frame_to_imu_align.front()->sensor_id = sensor_id;
                frame_to_imu_align.front()->ExT_ = ex_pose[sensor_id];
                for (int i = 1; i < window_frames.size();) {
                    Frame::Ptr new_frame(new Frame());
                    for (int j = 0; j < push_cout && i < window_frames.size();
                         j++) {
                        auto start_frame = window_frames.begin();
                        std::advance(start_frame, i);
                        Frame::Ptr curr_frame = *(start_frame);

                        new_frame->timeStamp = curr_frame->timeStamp;
                        new_frame->imuIntegrator.PushIMUMsg(
                            curr_frame->imuIntegrator.GetIMUMsg());
                        new_frame->P_ = T[i];
                        new_frame->Q_ = Q[i];
                        new_frame->sensor_id = sensor_id;
                        new_frame->ExT_ = ex_pose[sensor_id];
                        i++;
                    }
                    frame_to_imu_align.push_back(new_frame);
                }

                double scale = 1.0;
                Eigen::Vector3d GVector;
                if (tryImuAlignment(frame_to_imu_align, scale, GVector)) {
                    return 2;
                }
            }
        }
    }

    return 0;
}

bool CameraModule::relativePose(const std::string& cam_id,
                                Eigen::Matrix3d& relative_R,
                                Eigen::Vector3d& relative_T, int& l) {
    // find previous frame which contians enough correspondance and parallex
    // with newest frame

    for (int i = 0; i < window_frames.size() - 1; i++) {
        auto start_frame = window_frames.begin();
        std::advance(start_frame, i);
        auto corres =
            f_manager.getCorresponds(cam_id, i, window_frames.size() - 1);
        if (corres.size() > 20) {
            double sum_parallax = 0;
            for (int j = 0; j < int(corres.size()); j++) {
                Eigen::Vector2d pts_0(corres[j].first(0) / corres[j].first(2),
                                      corres[j].first(1) / corres[j].first(2));
                Eigen::Vector2d pts_1(
                    corres[j].second(0) / corres[j].second(2),
                    corres[j].second(1) / corres[j].second(2));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }

            double average_parallax = sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 &&
                solveRelativeRT(corres, relative_R, relative_T)) {
                l = i;
                return true;
            }
        }
    }
    return false;
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
        frame_to_imu_align.front()->P_ = frame_to_imu_align.front()->P;
        frame_to_imu_align.front()->Q_ = frame_to_imu_align.front()->Q;
        frame_to_imu_align.front()->ExT_ =
            ex_pose[frame_to_imu_align.front()->sensor_id];
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

                new_frame->P_ = curr_frame->P;
                new_frame->Q_ = curr_frame->Q;
                new_frame->sensor_id = curr_frame->sensor_id;
                new_frame->ExT_ = ex_pose[curr_frame->sensor_id];
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

            Eigen::Vector3d Pwl = scale * (*cur_frame)->P;
            Eigen::Quaterniond Qwl = (*cur_frame)->Q;
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

int LidarModule::initialize() {
    if (staticInitialize())
        return InitializtionType::STATIC;

    if (dynamicInitialize())
        return InitializtionType::DYNAMIC;

    return InitializtionType::NONE;
}

void LidarModule::RemoveLidarDistortion(pcl::PointCloud<PointType>::Ptr& cloud,
                                        const Eigen::Matrix3d& dRlc,
                                        const Eigen::Vector3d& dtlc) {
    int PointsNum = cloud->points.size();
    for (int i = 0; i < PointsNum; i++) {
        Eigen::Vector3d startP;
        float s = cloud->points[i].normal_x;
        if (s == 1.0)
            continue;
        Eigen::Quaterniond qlc = Eigen::Quaterniond(dRlc).normalized();
        Eigen::Quaterniond delta_qlc =
            Eigen::Quaterniond::Identity().slerp(s, qlc).normalized();
        const Eigen::Vector3d delta_Plc = s * dtlc;
        startP =
            delta_qlc * Eigen::Vector3d(cloud->points[i].x, cloud->points[i].y,
                                        cloud->points[i].z) +
            delta_Plc;
        Eigen::Vector3d _po = dRlc.transpose() * (startP - dtlc);

        cloud->points[i].x = _po(0);
        cloud->points[i].y = _po(1);
        cloud->points[i].z = _po(2);
        cloud->points[i].normal_x = 1.0;
    }
}

double CameraModule::dataSynchronize(const double& timestamp) {
    if (timestamp > 0) {
        if (!curr_data) {
            if (data_queue->empty()) {
                sleep(1 * 1e-3);
                return 0;
            } else {
                data_queue->pop(curr_data);
            }
        }

        while (curr_data->timestamp <= timestamp) {
            CameraFrame::Ptr new_frame(new CameraFrame);
            new_frame->trackResult = curr_data;
            new_frame->timeStamp = curr_data->timestamp;
            curr_timestamp = curr_data->timestamp;

            f_manager.addFeature(frame_count++, curr_data->observations, 0.0);

            // Debug for Show
            {
                const auto track_result = f_manager.showFrame(frame_count - 1);
                for (const auto& im : track_result) {
                    const std::string& cam_id = im.first;
                    if (curr_data->input_images->img_data.find(cam_id) !=
                        curr_data->input_images->img_data.end())
                        cv::imshow(
                            cam_id + "_track_result",
                            utility::drawKeypoint1(
                                curr_data->input_images->img_data[cam_id],
                                im.second));
                }
                cv::waitKey(10);
            }

            std::vector<ImuData::Ptr> imu_meas;
            getPreIntegratedImuData(imu_meas);
            if (imu_meas.size()) {
                new_frame->pre_imu_enabled = true;
                new_frame->imuIntegrator.PushIMUMsg(imu_meas);
                new_frame->imuIntegrator.PreIntegration(
                    last_timestamp, curr_timestamp, prev_frame->ba,
                    prev_frame->bg);
            } else {
                new_frame->pre_imu_enabled = false;
            }

            window_frames.push_back(new_frame);

            last_timestamp = curr_timestamp;

            if (!data_queue->empty()) {
                data_queue->pop(curr_data);
            } else {
                curr_data = nullptr;
                break;
            }
        }
    } else {
        if (data_queue->empty()) {
            sleep(1 * 1e-3);
            return 0;
        } else {
            data_queue->pop(curr_data);
        }

        CameraFrame::Ptr new_frame(new CameraFrame);
        new_frame->trackResult = curr_data;
        new_frame->timeStamp = curr_data->timestamp;
        curr_timestamp = curr_data->timestamp;

        f_manager.addFeature(frame_count++, curr_data->observations, 0.0);

        {
            const auto track_result = f_manager.showFrame(frame_count - 1);
            for (const auto& im : track_result) {
                const std::string& cam_id = im.first;
                if (curr_data->input_images->img_data.find(cam_id) !=
                    curr_data->input_images->img_data.end())
                    cv::imshow(cam_id + "_track_result",
                               utility::drawKeypoint1(
                                   curr_data->input_images->img_data[cam_id],
                                   im.second));
            }
            cv::waitKey(10);
        }

        std::vector<ImuData::Ptr> imu_meas;
        getPreIntegratedImuData(imu_meas);
        if (imu_meas.size()) {
            new_frame->pre_imu_enabled = true;
            new_frame->imuIntegrator.PushIMUMsg(imu_meas);
            new_frame->imuIntegrator.PreIntegration(
                last_timestamp, curr_timestamp, prev_frame->ba, prev_frame->bg);
        } else {
            new_frame->pre_imu_enabled = false;
        }

        window_frames.push_back(new_frame);

        last_timestamp = curr_timestamp;
    }

    return last_timestamp;
}

double LidarModule::dataSynchronize(const double& timestamp) {
    if (timestamp > 0) {
        if (!curr_data) {
            if (data_queue->empty()) {
                sleep(1 * 1e-3);
                return 0;
            } else {
                data_queue->pop(curr_data);
            }
        }

        while (curr_data->timestamp <= timestamp) {
            TicToc t0;

            LidarFrame::Ptr new_frame(new LidarFrame);
            new_frame->laserCloud = curr_data->features;
            new_frame->timeStamp = curr_data->timestamp;
            new_frame->sensor_id = curr_data->sensor_id;
            curr_timestamp = curr_data->timestamp;

            // viewer.addPointCloud(utility::convertToRGB(*(curr_data->features)),
            //                      "feat");

            Eigen::Matrix3d delta_Rl = Eigen::Matrix3d::Identity();
            Eigen::Vector3d delta_tl = Eigen::Vector3d::Zero();
            Eigen::Matrix4d exTbl = ex_pose[new_frame->sensor_id];
            Eigen::Matrix4d exTlb = exTbl.inverse();

            std::vector<ImuData::Ptr> imu_meas;
            getPreIntegratedImuData(imu_meas);
            if (imu_meas.size()) {
                if (imu_initialized) {
                    new_frame->pre_imu_enabled = true;
                    new_frame->imuIntegrator.PushIMUMsg(imu_meas);
                    new_frame->imuIntegrator.PreIntegration(
                        last_timestamp, curr_timestamp, prev_frame->ba,
                        prev_frame->bg);

                    const Eigen::Quaterniond& dQ =
                        new_frame->imuIntegrator.GetDeltaQ();
                    const Eigen::Vector3d& dP =
                        new_frame->imuIntegrator.GetDeltaP();
                    const Eigen::Vector3d& dV =
                        new_frame->imuIntegrator.GetDeltaV();
                    double dt = new_frame->imuIntegrator.GetDeltaTime();

                    const Eigen::Vector3d& Pwbpre = prev_frame->P;
                    const Eigen::Quaterniond& Qwbpre = prev_frame->Q;
                    const Eigen::Vector3d& Vwbpre = prev_frame->V;

                    new_frame->Q = Qwbpre * dQ;
                    new_frame->P = Pwbpre + Vwbpre * dt +
                                   0.5 * gravity_vector * dt * dt +
                                   Qwbpre * (dP);
                    new_frame->V = Vwbpre + gravity_vector * dt + Qwbpre * (dV);
                    new_frame->bg = prev_frame->bg;
                    new_frame->ba = prev_frame->ba;

                    Eigen::Quaterniond Qwlpre =
                        Qwbpre * Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
                    Eigen::Vector3d Pwlpre =
                        Qwbpre * exTbl.block<3, 1>(0, 3) + Pwbpre;

                    Eigen::Quaterniond Qwl =
                        new_frame->Q *
                        Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
                    Eigen::Vector3d Pwl =
                        new_frame->Q * exTbl.block<3, 1>(0, 3) + new_frame->P;

                    delta_Rl = Qwlpre.conjugate() * Qwl;
                    delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
                } else {
                    new_frame->pre_imu_enabled = true;
                    new_frame->imuIntegrator.PushIMUMsg(imu_meas);
                    new_frame->imuIntegrator.GyroIntegration(
                        last_timestamp, curr_timestamp, prev_frame->bg);

                    const Eigen::Quaterniond& dQ =
                        new_frame->imuIntegrator.GetDeltaQ();
                    const Eigen::Matrix3d& dR = dQ.toRotationMatrix();
                    const double dt = curr_timestamp - last_timestamp;

                    const Eigen::Vector3d& Pwbpre = prev_frame->P;
                    const Eigen::Quaterniond& Qwbpre = prev_frame->Q;

                    Eigen::Vector3d delta_tb = velocity.segment<3>(0) * dt;

                    // predict current lidar pose
                    new_frame->P = prev_frame->Q.toRotationMatrix() * delta_tb +
                                   prev_frame->P;
                    new_frame->Q = prev_frame->Q.toRotationMatrix() * dR;

                    // new_frame->P = prev_frame->P;
                    // new_frame->Q = prev_frame->Q;

                    Eigen::Quaterniond Qwlpre =
                        Qwbpre * Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
                    Eigen::Vector3d Pwlpre =
                        Qwbpre * exTbl.block<3, 1>(0, 3) + Pwbpre;

                    Eigen::Quaterniond Qwl =
                        new_frame->Q *
                        Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
                    Eigen::Vector3d Pwl =
                        new_frame->Q * exTbl.block<3, 1>(0, 3) + new_frame->P;

                    delta_Rl = Qwlpre.conjugate() * Qwl;
                    delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
                }
            } else {
                new_frame->pre_imu_enabled = false;

                const double dt = curr_timestamp - last_timestamp;

                const Eigen::Vector3d& Pwbpre = prev_frame->P;
                const Eigen::Quaterniond& Qwbpre = prev_frame->Q;

                Eigen::Vector3d delta_tb = velocity.segment<3>(0) * dt;
                Eigen::Matrix3d delta_Rb =
                    (Sophus::SO3d::exp(velocity.segment<3>(3) * dt)
                         .unit_quaternion())
                        .toRotationMatrix();

                new_frame->P =
                    prev_frame->Q.toRotationMatrix() * delta_tb + prev_frame->P;
                new_frame->Q = prev_frame->Q.toRotationMatrix() * delta_Rb;

                // new_frame->P = prev_frame->P;
                // new_frame->Q = prev_frame->Q;

                Eigen::Quaterniond Qwlpre =
                    Qwbpre * Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
                Eigen::Vector3d Pwlpre =
                    Qwbpre * exTbl.block<3, 1>(0, 3) + Pwbpre;

                Eigen::Quaterniond Qwl =
                    new_frame->Q * Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
                Eigen::Vector3d Pwl =
                    new_frame->Q * exTbl.block<3, 1>(0, 3) + new_frame->P;

                delta_Rl = Qwlpre.conjugate() * Qwl;
                delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
            }

            logger->recordLogger(logger_flag, t0.toc(), frame_count,
                                 "dataSynchronize | Part0");
            t0.tic();

            RemoveLidarDistortion(new_frame->laserCloud, delta_Rl, delta_tl);

            logger->recordLogger(
                logger_flag, t0.toc(), frame_count,
                "dataSynchronize RemoveLidarDistortion | Part1");
            t0.tic();

            window_frames.push_back(new_frame);
            frame_count++;

            if (!imu_initialized) {
                getBackLidarPose();
            }

            if (last_timestamp > 0) {
                velocity.segment<3>(3) =
                    (prev_frame->Q.inverse() * (new_frame->P - prev_frame->P)) /
                    (curr_timestamp - last_timestamp);
                velocity.segment<3>(0) =
                    Sophus::SO3d(prev_frame->Q.inverse() * new_frame->Q).log();
                velocity.segment<3>(0) =
                    velocity.segment<3>(0) / (curr_timestamp - last_timestamp);
            }
            prev_frame = new_frame;
            last_timestamp = curr_timestamp;

            logger->recordLogger(logger_flag, t0.toc(), frame_count,
                                 "dataSynchronize getBackLidarPose | Part2");
            t0.tic();

            if (!data_queue->empty()) {
                data_queue->pop(curr_data);
            } else {
                curr_data = nullptr;
                break;
            }
        }
    } else {
        if (data_queue->empty()) {
            sleep(1 * 1e-3);
            return 0;
        } else {
            data_queue->pop(curr_data);
        }

        TicToc t0;

        LidarFrame::Ptr new_frame(new LidarFrame);
        new_frame->laserCloud = curr_data->features;
        new_frame->timeStamp = curr_data->timestamp;
        new_frame->sensor_id = curr_data->sensor_id;
        curr_timestamp = curr_data->timestamp;

        // viewer.addPointCloud(utility::convertToRGB(*(curr_data->features)),
        //                      "feat");

        Eigen::Matrix3d delta_Rl = Eigen::Matrix3d::Identity();
        Eigen::Vector3d delta_tl = Eigen::Vector3d::Zero();
        Eigen::Matrix4d exTbl = ex_pose[new_frame->sensor_id];
        Eigen::Matrix4d exTlb = exTbl.inverse();

        std::vector<ImuData::Ptr> imu_meas;
        getPreIntegratedImuData(imu_meas);
        if (imu_meas.size()) {
            if (imu_initialized) {
                new_frame->pre_imu_enabled = true;
                new_frame->imuIntegrator.PushIMUMsg(imu_meas);
                new_frame->imuIntegrator.PreIntegration(
                    last_timestamp, curr_timestamp, prev_frame->ba,
                    prev_frame->bg);

                const Eigen::Quaterniond& dQ =
                    new_frame->imuIntegrator.GetDeltaQ();
                const Eigen::Vector3d& dP =
                    new_frame->imuIntegrator.GetDeltaP();
                const Eigen::Vector3d& dV =
                    new_frame->imuIntegrator.GetDeltaV();
                double dt = new_frame->imuIntegrator.GetDeltaTime();

                const Eigen::Vector3d& Pwbpre = prev_frame->P;
                const Eigen::Quaterniond& Qwbpre = prev_frame->Q;
                const Eigen::Vector3d& Vwbpre = prev_frame->V;

                new_frame->Q = Qwbpre * dQ;
                new_frame->P = Pwbpre + Vwbpre * dt +
                               0.5 * gravity_vector * dt * dt + Qwbpre * (dP);
                new_frame->V = Vwbpre + gravity_vector * dt + Qwbpre * (dV);
                new_frame->bg = prev_frame->bg;
                new_frame->ba = prev_frame->ba;

                Eigen::Quaterniond Qwlpre =
                    Qwbpre * Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
                Eigen::Vector3d Pwlpre =
                    Qwbpre * exTbl.block<3, 1>(0, 3) + Pwbpre;

                Eigen::Quaterniond Qwl =
                    new_frame->Q * Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
                Eigen::Vector3d Pwl =
                    new_frame->Q * exTbl.block<3, 1>(0, 3) + new_frame->P;

                delta_Rl = Qwlpre.conjugate() * Qwl;
                delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
            } else {
                new_frame->pre_imu_enabled = true;
                new_frame->imuIntegrator.PushIMUMsg(imu_meas);
                new_frame->imuIntegrator.GyroIntegration(
                    last_timestamp, curr_timestamp, prev_frame->bg);

                const Eigen::Quaterniond& dQ =
                    new_frame->imuIntegrator.GetDeltaQ();
                const Eigen::Matrix3d& dR = dQ.toRotationMatrix();
                const double dt = curr_timestamp - last_timestamp;

                const Eigen::Vector3d& Pwbpre = prev_frame->P;
                const Eigen::Quaterniond& Qwbpre = prev_frame->Q;

                Eigen::Vector3d delta_tb = velocity.segment<3>(0) * dt;

                // predict current lidar pose
                new_frame->P =
                    prev_frame->Q.toRotationMatrix() * delta_tb + prev_frame->P;
                new_frame->Q = prev_frame->Q.toRotationMatrix() * dR;

                // new_frame->P = prev_frame->P;
                // new_frame->Q = prev_frame->Q;

                Eigen::Quaterniond Qwlpre =
                    Qwbpre * Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
                Eigen::Vector3d Pwlpre =
                    Qwbpre * exTbl.block<3, 1>(0, 3) + Pwbpre;

                Eigen::Quaterniond Qwl =
                    new_frame->Q * Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
                Eigen::Vector3d Pwl =
                    new_frame->Q * exTbl.block<3, 1>(0, 3) + new_frame->P;

                delta_Rl = Qwlpre.conjugate() * Qwl;
                delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
            }
        } else {
            new_frame->pre_imu_enabled = false;

            const double dt = curr_timestamp - last_timestamp;

            const Eigen::Vector3d& Pwbpre = prev_frame->P;
            const Eigen::Quaterniond& Qwbpre = prev_frame->Q;

            Eigen::Vector3d delta_tb = velocity.segment<3>(0) * dt;
            Eigen::Matrix3d delta_Rb =
                (Sophus::SO3d::exp(velocity.segment<3>(3) * dt)
                     .unit_quaternion())
                    .toRotationMatrix();

            new_frame->P =
                prev_frame->Q.toRotationMatrix() * delta_tb + prev_frame->P;
            new_frame->Q = prev_frame->Q.toRotationMatrix() * delta_Rb;

            // new_frame->P = prev_frame->P;
            // new_frame->Q = prev_frame->Q;

            Eigen::Quaterniond Qwlpre =
                Qwbpre * Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
            Eigen::Vector3d Pwlpre = Qwbpre * exTbl.block<3, 1>(0, 3) + Pwbpre;

            Eigen::Quaterniond Qwl =
                new_frame->Q * Eigen::Quaterniond(exTbl.block<3, 3>(0, 0));
            Eigen::Vector3d Pwl =
                new_frame->Q * exTbl.block<3, 1>(0, 3) + new_frame->P;

            delta_Rl = Qwlpre.conjugate() * Qwl;
            delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
        }

        logger->recordLogger(logger_flag, t0.toc(), frame_count,
                             "dataSynchronize | Part0");
        t0.tic();

        RemoveLidarDistortion(new_frame->laserCloud, delta_Rl, delta_tl);

        logger->recordLogger(logger_flag, t0.toc(), frame_count,
                             "dataSynchronize RemoveLidarDistortion | Part1");
        t0.tic();

        window_frames.push_back(new_frame);
        frame_count++;

        if (!imu_initialized) {
            getBackLidarPose();
        }

        logger->recordLogger(logger_flag, t0.toc(), frame_count,
                             "dataSynchronize getBackLidarPose | Part2");
        t0.tic();

        if (last_timestamp > 0) {
            velocity.segment<3>(3) =
                (prev_frame->Q.inverse() * (new_frame->P - prev_frame->P)) /
                (curr_timestamp - last_timestamp);
            velocity.segment<3>(0) =
                Sophus::SO3d(prev_frame->Q.inverse() * new_frame->Q).log();
            velocity.segment<3>(0) =
                velocity.segment<3>(0) / (curr_timestamp - last_timestamp);
        }
        prev_frame = new_frame;
        last_timestamp = curr_timestamp;
    }

    return last_timestamp;
}

void LidarModule::getBackLidarPose() {
    TicToc t0;

    auto frame_curr = window_frames.back();
    std::string lidar_id = frame_curr->sensor_id;
    Eigen::Matrix<double, 3, 3> exRbl = ex_pose[lidar_id].block<3, 3>(0, 0);
    Eigen::Matrix<double, 3, 1> exPbl = ex_pose[lidar_id].block<3, 1>(0, 3);

    int laserCloudCornerFromMapNum =
        map_manager->get_corner_map()->points.size();
    int laserCloudSurfFromMapNum = map_manager->get_surf_map()->points.size();
    int laserCloudCornerFromLocalNum = laserCloudCornerFromLocal->points.size();
    int laserCloudSurfFromLocalNum = laserCloudSurfFromLocal->points.size();

    laserCloudCornerLast[0]->clear();
    laserCloudSurfLast[0]->clear();
    laserCloudNonFeatureLast[0]->clear();

    const auto& laserCloud_curr =
        std::dynamic_pointer_cast<LidarFrame>(frame_curr)->laserCloud;
    for (const auto& p : laserCloud_curr->points) {
        if (std::fabs(p.normal_z - 1.0) < 1e-5)
            laserCloudCornerLast[0]->push_back(p);
        else if (std::fabs(p.normal_z - 2.0) < 1e-5)
            laserCloudSurfLast[0]->push_back(p);
        else if (std::fabs(p.normal_z - 3.0) < 1e-5)
            laserCloudNonFeatureLast[0]->push_back(p);
    }

    laserCloudCornerStack[0]->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast[0]);
    downSizeFilterCorner.filter(*laserCloudCornerStack[0]);

    laserCloudSurfStack[0]->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast[0]);
    downSizeFilterSurf.filter(*laserCloudSurfStack[0]);

    laserCloudNonFeatureStack[0]->clear();
    downSizeFilterNonFeature.setInputCloud(laserCloudNonFeatureLast[0]);
    downSizeFilterNonFeature.filter(*laserCloudNonFeatureStack[0]);

    logger->recordLogger(logger_flag, t0.toc(), frame_count,
                         "getBackLidarPose | Part0");
    t0.tic();

    if (((laserCloudCornerFromMapNum > 0 && laserCloudSurfFromMapNum > 100) ||
         (laserCloudCornerFromLocalNum > 0 &&
          laserCloudSurfFromLocalNum > 100))) {
        if (0) {
            auto frame_last = std::dynamic_pointer_cast<LidarFrame>(prev_frame);
            gicpScan2Scan.setInputTarget(frame_last->laserCloud);
            pcl::PointCloud<PointType>::Ptr source(
                new pcl::PointCloud<PointType>);
            *source = *laserCloudSurfStack[0];
            gicpScan2Scan.setInputSource(source);
            pcl::PointCloud<PointType>::Ptr aligned(
                new pcl::PointCloud<PointType>);
            gicpScan2Scan.align(*aligned);

            Eigen::Matrix4f Twl_last = Eigen::Matrix4f::Identity();
            Twl_last.block<3, 3>(0, 0) =
                frame_last->Q.toRotationMatrix().cast<float>();
            Twl_last.block<3, 1>(0, 3) = frame_last->P.cast<float>();

            Eigen::Matrix4f Twl_final =
                Twl_last * gicpScan2Scan.getFinalTransformation();
            frame_curr->Q =
                Eigen::Quaterniond(Twl_final.block<3, 3>(0, 0).cast<double>());
            frame_curr->P = Twl_final.block<3, 1>(0, 3).cast<double>();
        } else if (1) {
            pcl::PointCloud<PointType>::Ptr laserCloudCornerFromLocalDS(
                new pcl::PointCloud<PointType>);
            downSizeFilterCorner.setInputCloud(laserCloudCornerFromLocal);
            downSizeFilterCorner.filter(*laserCloudCornerFromLocalDS);

            pcl::PointCloud<PointType>::Ptr laserCloudSurfFromLocalDS(
                new pcl::PointCloud<PointType>);
            downSizeFilterSurf.setInputCloud(laserCloudSurfFromLocal);
            downSizeFilterSurf.filter(*laserCloudSurfFromLocalDS);

            pcl::PointCloud<PointType>::Ptr localMap(
                new pcl::PointCloud<PointType>);
            *localMap += *laserCloudCornerFromLocal;
            *localMap += *laserCloudSurfFromLocal;

            pcl::PointCloud<PointType>::Ptr currScan(
                new pcl::PointCloud<PointType>);
            *currScan += *laserCloudSurfStack[0];
            *currScan += *laserCloudCornerStack[0];

            gicpScan2Map.setInputTarget(localMap);
            gicpScan2Map.setInputSource(currScan);
            pcl::PointCloud<PointType>::Ptr aligned(
                new pcl::PointCloud<PointType>);
            Eigen::Matrix4f Twl_init = Eigen::Matrix4f::Identity();
            Twl_init.block<3, 3>(0, 0) =
                frame_curr->Q.toRotationMatrix().cast<float>() *
                exRbl.cast<float>();
            Twl_init.block<3, 1>(0, 3) =
                frame_curr->Q.cast<float>() * exPbl.cast<float>() +
                frame_curr->P.cast<float>();
            gicpScan2Map.align(*aligned, Twl_init);

            Eigen::Matrix4f Twl_final = gicpScan2Map.getFinalTransformation();
            frame_curr->Q = Eigen::Quaterniond(
                Twl_final.block<3, 3>(0, 0).cast<double>() * exRbl.transpose());
            frame_curr->P = -frame_curr->Q.toRotationMatrix() * exPbl +
                            Twl_final.block<3, 1>(0, 3).cast<double>();
        } else {
            kdtreeCornerFromLocal->setInputCloud(laserCloudCornerFromLocal);
            kdtreeSurfFromLocal->setInputCloud(laserCloudSurfFromLocal);
            kdtreeNonFeatureFromLocal->setInputCloud(
                laserCloudNonFeatureFromLocal);

            std::unique_lock<std::mutex> locker3(map_manager->mtx_MapManager);
            for (int i = 0; i < 4851; i++) {
                CornerKdMap[i] = map_manager->getCornerKdMap(i);
                SurfKdMap[i] = map_manager->getSurfKdMap(i);
                NonFeatureKdMap[i] = map_manager->getNonFeatureKdMap(i);

                GlobalSurfMap[i] = map_manager->laserCloudSurf_for_match[i];
                GlobalCornerMap[i] = map_manager->laserCloudCorner_for_match[i];
                GlobalNonFeatureMap[i] =
                    map_manager->laserCloudNonFeature_for_match[i];
            }
            laserCenWidth_last = map_manager->get_laserCloudCenWidth_last();
            laserCenHeight_last = map_manager->get_laserCloudCenHeight_last();
            laserCenDepth_last = map_manager->get_laserCloudCenDepth_last();
            locker3.unlock();

            // store point to line features
            vLineFeatures.clear();
            vLineFeatures.resize(1);
            vLineFeatures[0].reserve(2000);

            vPlanFeatures.clear();
            vPlanFeatures.resize(1);
            vPlanFeatures[0].reserve(2000);

            vNonFeatures.clear();
            vNonFeatures.resize(1);
            vNonFeatures[0].reserve(2000);

            plan_weight_tan = 0.0;
            thres_dist = 25.0;

            const int max_iters = 6;
            for (int iterOpt = 0; iterOpt < max_iters; iterOpt++) {
                vector2double();

                q_before_opti = frame_curr->Q;
                t_before_opti = frame_curr->P;

                std::vector<std::vector<ceres::CostFunction*>> edgesLine(1);
                std::vector<std::vector<ceres::CostFunction*>> edgesPlan(1);
                std::vector<std::vector<ceres::CostFunction*>> edgesNon(1);

                Eigen::Matrix4d transformTobeMapped =
                    Eigen::Matrix4d::Identity();
                transformTobeMapped.topLeftCorner(3, 3) = frame_curr->Q * exRbl;
                transformTobeMapped.topRightCorner(3, 1) =
                    frame_curr->Q * exPbl + frame_curr->P;

                std::thread threads[3];
                threads[0] = std::thread(&LidarModule::processPointToLine, this,
                                         std::ref(edgesLine[0]),
                                         std::ref(vLineFeatures[0]),
                                         std::ref(laserCloudCornerStack[0]),
                                         std::ref(laserCloudCornerFromLocal),
                                         std::ref(kdtreeCornerFromLocal),
                                         std::ref(transformTobeMapped));

                threads[1] = std::thread(&LidarModule::processPointToPlan, this,
                                         std::ref(edgesPlan[0]),
                                         std::ref(vPlanFeatures[0]),
                                         std::ref(laserCloudSurfStack[0]),
                                         std::ref(laserCloudSurfFromLocal),
                                         std::ref(kdtreeSurfFromLocal),
                                         std::ref(transformTobeMapped));

                threads[2] = std::thread(
                    &LidarModule::processNonFeatureICP, this,
                    std::ref(edgesNon[0]), std::ref(vNonFeatures[0]),
                    std::ref(laserCloudNonFeatureStack[0]),
                    std::ref(laserCloudNonFeatureFromLocal),
                    std::ref(kdtreeNonFeatureFromLocal),
                    std::ref(transformTobeMapped));

                threads[0].join();
                threads[1].join();
                threads[2].join();

                int window_size = window_frames.size();

                ceres::Problem init_problem;

                ceres::LocalParameterization* local_parameterization1 = NULL;
                init_problem.AddParameterBlock(para_Pose[window_size - 1],
                                               SIZE_POSE,
                                               local_parameterization1);

                for (auto c : para_Ex_Pose) {
                    ceres::LocalParameterization* local_parameterization2 =
                        NULL;
                    init_problem.AddParameterBlock(c.second, SIZE_POSE,
                                                   local_parameterization2);
                    if (1) {
                        init_problem.SetParameterBlockConstant(c.second);
                    }
                }

                {
                    residual_block_ids.clear();
                    residual_block_ids["lidar_corner"] =
                        std::vector<ceres::ResidualBlockId>();
                    residual_block_ids["lidar_surf"] =
                        std::vector<ceres::ResidualBlockId>();
                    residual_block_ids["lidar_nonfeat"] =
                        std::vector<ceres::ResidualBlockId>();
                }

                // create huber loss function
                ceres::LossFunction* loss_function = NULL;
                loss_function =
                    new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);

                int cntSurf = 0;
                int cntCorner = 0;
                int cntNon = 0;

                thres_dist = 1.0;
                if (iterOpt == 0) {
                    thres_dist = 10.0;
                }

                int cntFtu = 0;
                for (auto& e : edgesLine[0]) {
                    if (std::fabs(vLineFeatures[0][cntFtu].error) > 1e-5) {
                        auto re_id = init_problem.AddResidualBlock(
                            e, loss_function, para_Pose[window_size - 1],
                            para_Ex_Pose[lidar_id]);
                        vLineFeatures[0][cntFtu].valid = true;
                        residual_block_ids["lidar_corner"].push_back(re_id);
                    } else {
                        vLineFeatures[0][cntFtu].valid = false;
                    }
                    cntFtu++;
                    cntCorner++;
                }

                cntFtu = 0;
                for (auto& e : edgesPlan[0]) {
                    if (std::fabs(vPlanFeatures[0][cntFtu].error) > 1e-5) {
                        auto re_id = init_problem.AddResidualBlock(
                            e, loss_function, para_Pose[window_size - 1],
                            para_Ex_Pose[lidar_id]);
                        vPlanFeatures[0][cntFtu].valid = true;
                        residual_block_ids["lidar_surf"].push_back(re_id);
                    } else {
                        vPlanFeatures[0][cntFtu].valid = false;
                    }
                    cntFtu++;
                    cntSurf++;
                }

                cntFtu = 0;
                for (auto& e : edgesNon[0]) {
                    if (std::fabs(vNonFeatures[0][cntFtu].error) > 1e-5) {
                        auto re_id = init_problem.AddResidualBlock(
                            e, loss_function, para_Pose[window_size - 1],
                            para_Ex_Pose[lidar_id]);
                        vNonFeatures[0][cntFtu].valid = true;
                        residual_block_ids["lidar_nonfeat"].push_back(re_id);
                    } else {
                        vNonFeatures[0][cntFtu].valid = false;
                    }
                    cntFtu++;
                    cntNon++;
                }

                // Before
                if (show_opt) {
                    std::cout << " cntCorner : " << cntCorner << std::endl;
                    std::cout << " cntSurf : " << cntSurf << std::endl;
                    std::cout << " cntNon : " << cntNon << std::endl
                              << std::endl;

                    std::cout << "Opt Before" << std::endl;

                    ceres::Problem::EvaluateOptions init_evaluate_options;
                    init_problem.GetParameterBlocks(
                        &(init_evaluate_options.parameter_blocks));

                    // module
                    for (auto t : residual_block_ids) {
                        if (t.second.size()) {
                            init_evaluate_options.residual_blocks = t.second;
                            double cost;
                            std::vector<double> residuals;
                            init_problem.Evaluate(init_evaluate_options, &cost,
                                                  &residuals, NULL, NULL);

                            std::cout << t.first << " : " << std::fixed
                                      << std::setprecision(6) << cost << " | ";
                            // for (int j = 0; j < residuals.size(); j++)
                            //     std::cout << std::fixed <<
                            //     std::setprecision(6)
                            //               << residuals[j] << " ";
                            // std::cout << std::endl;
                            std::cout << std::endl;
                        }
                    }

                    double total_cost;
                    init_evaluate_options.residual_blocks.clear();
                    init_problem.Evaluate(init_evaluate_options, &total_cost,
                                          NULL, NULL, NULL);
                    std::cout << "total_cost"
                              << " : " << std::fixed << std::setprecision(6)
                              << total_cost << " | ";
                    std::cout << std::endl;
                }

                ceres::Solver::Options options;
                // options.max_solver_time_in_seconds = 0.02;
                options.linear_solver_type = ceres::DENSE_SCHUR;
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.max_num_iterations = 10;
                // options.minimizer_progress_to_stdout = false;
                options.minimizer_progress_to_stdout = true;
                // options.num_threads = 6;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &init_problem, &summary);
                std::cout << summary.FullReport() << std::endl;

                // After
                if (show_opt) {
                    std::cout << "Opt After" << std::endl;

                    ceres::Problem::EvaluateOptions init_evaluate_options;
                    init_problem.GetParameterBlocks(
                        &(init_evaluate_options.parameter_blocks));

                    // module
                    for (auto t : residual_block_ids) {
                        if (t.second.size()) {
                            init_evaluate_options.residual_blocks = t.second;
                            double cost;
                            std::vector<double> residuals;
                            init_problem.Evaluate(init_evaluate_options, &cost,
                                                  &residuals, NULL, NULL);

                            std::cout << t.first << " : " << std::fixed
                                      << std::setprecision(6) << cost << " | ";
                            // for (int j = 0; j < residuals.size(); j++)
                            //     std::cout << std::fixed <<
                            //     std::setprecision(6)
                            //               << residuals[j] << " ";
                            // std::cout << std::endl;
                            std::cout << std::endl;
                        }
                    }

                    double total_cost;
                    init_evaluate_options.residual_blocks.clear();
                    init_problem.Evaluate(init_evaluate_options, &total_cost,
                                          NULL, NULL, NULL);
                    std::cout << "total_cost"
                              << " : " << std::fixed << std::setprecision(6)
                              << total_cost << " | ";
                    std::cout << std::endl;
                }

                double2vector();

                Eigen::Quaterniond q_after_opti = frame_curr->Q;
                Eigen::Vector3d t_after_opti = frame_curr->P;
                Eigen::Vector3d V_after_opti = frame_curr->V;
                double deltaR = (q_before_opti.angularDistance(q_after_opti)) *
                                180.0 / M_PI;
                double deltaT = (t_before_opti - t_after_opti).norm();

                if (deltaR < 0.05 && deltaT < 0.05 ||
                    (iterOpt + 1) == max_iters) {
                    break;
                }
            }
        }
    }

    logger->recordLogger(logger_flag, t0.toc(), frame_count,
                         "getBackLidarPose Optimaizition | Part1");
    t0.tic();

    Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
    transformTobeMapped.topLeftCorner(3, 3) = frame_curr->Q * exRbl;
    transformTobeMapped.topRightCorner(3, 1) =
        frame_curr->Q * exPbl + frame_curr->P;

    std::unique_lock<std::mutex> locker(mtx_Map);
    *laserCloudCornerForMap = *laserCloudCornerStack[0];
    *laserCloudSurfForMap = *laserCloudSurfStack[0];
    *laserCloudNonFeatureForMap = *laserCloudNonFeatureStack[0];
    transformForMap = transformTobeMapped;
    MapIncrementLocal(laserCloudCornerForMap, laserCloudSurfForMap,
                      laserCloudNonFeatureForMap, transformForMap);

    {
        map_viewer.addLocalMap(laserCloudCornerFromLocal, "local_corner", 110,
                               0, 0);
        map_viewer.addLocalMap(laserCloudSurfFromLocal, "local_surf", 0, 110,
                               0);
        size_t Id = (localMapID - 1) % localMapWindowSize;
        map_viewer.addCurrPoints(localCornerMap[Id], "curr_corner", 200, 0, 0);
        map_viewer.addCurrPoints(localSurfMap[Id], "curr_surf", 0, 200, 0);
        map_viewer.addPose(frame_curr->timeStamp, transformTobeMapped);
    }
    locker.unlock();

    logger->recordLogger(logger_flag, t0.toc(), frame_count,
                         "getBackLidarPose MapIncrementLocal | Part2");
    t0.tic();

    // Debug For Viewer
    {
        // pcl::PointCloud<PointType>::Ptr laserCloudAfterEstimate(
        //     new pcl::PointCloud<PointType>());
        // for (int i = 0; i < frame_curr->laserCloud->points.size(); i++) {
        //     PointType temp_point;
        //     MAP_MANAGER::pointAssociateToMap(
        //         &frame_curr->laserCloud->points[i], &temp_point,
        //         transformTobeMapped);
        //     laserCloudAfterEstimate->push_back(temp_point);
        // }
        // viewer.addPointCloud(
        //     utility::convertToRGB(*laserCloudCornerFromLocal),
        //     "local_corner");
        // viewer.addPointCloud(
        //     utility::convertToRGB(*laserCloudSurfFromLocal),
        //     "local_surf");
    }
}

void CameraModule::preProcess() {}

void CameraModule::postProcess() {}

void CameraModule::slideWindow(const Module::Ptr& prime_module) {}

void CameraModule::vector2double() {
    for (int i = 0; i < window_frames.size(); i++) {
        auto cf = window_frames.begin();
        std::advance(cf, i);
        Eigen::Map<Eigen::Matrix<double, 6, 1>> PR(para_Pose[i]);
        PR.segment<3>(0) = (*cf)->P;
        PR.segment<3>(3) = Sophus::SO3d((*cf)->Q).log();

        Eigen::Map<Eigen::Matrix<double, 9, 1>> VBias(para_SpeedBias[i]);
        VBias.segment<3>(0) = (*cf)->V;
        VBias.segment<3>(3) = (*cf)->bg;
        VBias.segment<3>(6) = (*cf)->ba;
    }

    for (auto c : ex_pose) {
        if (para_Ex_Pose.find(c.first) == para_Ex_Pose.end())
            para_Ex_Pose[c.first] = new double[SIZE_POSE];
        Eigen::Map<Eigen::Matrix<double, 6, 1>> Exbc(para_Ex_Pose[c.first]);
        const auto& exTbc = c.second;
        Exbc.segment<3>(0) = exTbc.block<3, 1>(0, 3);
        Exbc.segment<3>(3) =
            Sophus::SO3d(Eigen::Quaterniond(exTbc.block<3, 3>(0, 0))
                             .normalized()
                             .toRotationMatrix())
                .log();
    }

    auto dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
}

void CameraModule::addParameter() {}

void CameraModule::double2vector() {
    for (auto c : ex_pose) {
        Eigen::Map<Eigen::Matrix<double, 6, 1>> Exbc(para_Ex_Pose[c.first]);
        c.second = Eigen::Matrix4d::Identity();
        c.second.block<3, 3>(0, 0) =
            (Sophus::SO3d::exp(Exbc.segment<3>(3)).unit_quaternion())
                .toRotationMatrix();
        c.second.block<3, 1>(0, 3) = Exbc.segment<3>(0);
    }

    for (int i = 0; i < window_frames.size(); i++) {
        auto cf = window_frames.begin();
        std::advance(cf, i);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> PR(para_Pose[i]);
        Eigen::Map<const Eigen::Matrix<double, 9, 1>> VBias(para_SpeedBias[i]);
        (*cf)->P = PR.segment<3>(0);
        (*cf)->Q = Sophus::SO3d::exp(PR.segment<3>(3)).unit_quaternion();
        (*cf)->V = VBias.segment<3>(0);
        (*cf)->bg = VBias.segment<3>(3);
        (*cf)->ba = VBias.segment<3>(6);
        (*cf)->ExT_ = ex_pose[(*cf)->sensor_id];
    }

    auto dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
}

void CameraModule::addResidualBlock(int iterOpt) {
    ceres::LossFunction* loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);

    for (int i = 0; i < window_frames.size(); i++) {
        ceres::LocalParameterization* local_parameterization = NULL;
        problem->AddParameterBlock(para_Pose[i], SIZE_POSE,
                                   local_parameterization);
        problem->AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    for (auto c : para_Ex_Pose) {
        ceres::LocalParameterization* local_parameterization = NULL;
        problem->AddParameterBlock(c.second, SIZE_POSE, local_parameterization);
        if (1) {
            problem->SetParameterBlockConstant(c.second);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto& it_per_id : f_manager.feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].points[0];
        auto cam_id0 = it_per_id.feature_per_frame[0].camids[0];
        for (auto& it_per_frame : it_per_id.feature_per_frame) {
            imu_j++;
            if (imu_i != imu_j) {
                Eigen::Vector3d pts_j = it_per_frame.points[0];
                problem->AddResidualBlock(
                    ProjectionTwoFrameOneCamFactor::Create(
                        pts_i, pts_j, Eigen::Matrix2d::Identity()),
                    loss_function, para_Pose[imu_i], para_Pose[imu_j],
                    para_Ex_Pose[cam_id0], para_Feature[feature_index]);
            }

            if (it_per_frame.is_stereo) {
                for (size_t i = 1; i < it_per_frame.points.size(); i++) {
                    Eigen::Vector3d pts_j_right = it_per_frame.points[i];
                    auto cam_id1 = it_per_frame.camids[i];
                    if (imu_i != imu_j) {
                        problem->AddResidualBlock(
                            ProjectionTwoFrameTwoCamFactor::Create(
                                pts_i, pts_j_right,
                                Eigen::Matrix2d::Identity()),
                            loss_function, para_Pose[imu_i], para_Pose[imu_j],
                            para_Ex_Pose[cam_id0], para_Ex_Pose[cam_id1],
                            para_Feature[feature_index]);
                    } else {
                        problem->AddResidualBlock(
                            ProjectionOneFrameTwoCamFactor::Create(
                                pts_i, pts_j_right,
                                Eigen::Matrix2d::Identity()),
                            loss_function, para_Ex_Pose[cam_id0],
                            para_Ex_Pose[cam_id1], para_Feature[feature_index]);
                    }
                }
            }
            f_m_cnt++;
        }
    }
}

void CameraModule::marginalization1(
    MarginalizationInfo* last_marginalization_info,
    std::vector<double*>& last_marginalization_parameter_blocks,
    MarginalizationInfo* marginalization_info, int slide_win_size) {
    ceres::LossFunction* loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);

    int feature_index = -1;
    for (auto& it_per_id : f_manager.feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        if (imu_i >= slide_win_size)
            continue;

        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].points[0];
        auto cam_id0 = it_per_id.feature_per_frame[0].camids[0];

        for (auto& it_per_frame : it_per_id.feature_per_frame) {
            imu_j++;
            if (imu_i != imu_j) {
                Eigen::Vector3d pts_j = it_per_frame.points[0];
                ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
                    ProjectionTwoFrameOneCamFactor::Create(
                        pts_i, pts_j, Eigen::Matrix2d::Identity()),
                    loss_function,
                    std::vector<double*>{para_Pose[imu_i], para_Pose[imu_j],
                                         para_Ex_Pose[cam_id0],
                                         para_Feature[feature_index]},
                    std::vector<int>{0, 3});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            if (it_per_frame.is_stereo) {
                for (size_t i = 1; i < it_per_frame.points.size(); i++) {
                    Eigen::Vector3d pts_j_right = it_per_frame.points[i];
                    auto cam_id1 = it_per_frame.camids[i];
                    if (imu_i != imu_j) {
                        ResidualBlockInfo* residual_block_info =
                            new ResidualBlockInfo(
                                ProjectionTwoFrameTwoCamFactor::Create(
                                    pts_i, pts_j_right,
                                    Eigen::Matrix2d::Identity()),
                                loss_function,
                                std::vector<double*>{
                                    para_Pose[imu_i], para_Pose[imu_j],
                                    para_Ex_Pose[cam_id0],
                                    para_Ex_Pose[cam_id1],
                                    para_Feature[feature_index]},
                                std::vector<int>{0, 4});
                        marginalization_info->addResidualBlockInfo(
                            residual_block_info);
                    } else {
                        ResidualBlockInfo* residual_block_info =
                            new ResidualBlockInfo(
                                ProjectionOneFrameTwoCamFactor::Create(
                                    pts_i, pts_j_right,
                                    Eigen::Matrix2d::Identity()),
                                loss_function,
                                std::vector<double*>{
                                    para_Ex_Pose[cam_id0],
                                    para_Ex_Pose[cam_id1],
                                    para_Feature[feature_index]},
                                std::vector<int>{2});
                        marginalization_info->addResidualBlockInfo(
                            residual_block_info);
                    }
                }
            }
        }
    }
}

void CameraModule::marginalization2(
    std::unordered_map<long, double*>& addr_shift, int slide_win_size) {
    for (int i = slide_win_size; i < window_frames.size(); i++) {
        addr_shift[reinterpret_cast<long>(para_Pose[i])] =
            para_Pose[i - slide_win_size];
        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] =
            para_SpeedBias[i - slide_win_size];
    }
    for (auto c : para_Ex_Pose)
        addr_shift[reinterpret_cast<long>(c.second)] = c.second;
}

LidarModule::LidarModule() : Module() {
    sensorType = SensorFlag::LIDAR;
    prime_flag = false;
    scale = 0;

    laserCloudCornerFromLocal.reset(new pcl::PointCloud<PointType>);
    laserCloudSurfFromLocal.reset(new pcl::PointCloud<PointType>);
    laserCloudNonFeatureFromLocal.reset(new pcl::PointCloud<PointType>);
    initGroundCloud.reset(new pcl::PointCloud<PointType>);
    init_ground_count = 0;

    laserCloudCornerLast.resize(MAX_SLIDE_WINDOW_SIZE);
    for (auto& p : laserCloudCornerLast)
        p.reset(new pcl::PointCloud<PointType>);
    laserCloudSurfLast.resize(MAX_SLIDE_WINDOW_SIZE);
    for (auto& p : laserCloudSurfLast)
        p.reset(new pcl::PointCloud<PointType>);
    laserCloudNonFeatureLast.resize(MAX_SLIDE_WINDOW_SIZE);
    for (auto& p : laserCloudNonFeatureLast)
        p.reset(new pcl::PointCloud<PointType>);
    laserCloudCornerStack.resize(MAX_SLIDE_WINDOW_SIZE);
    for (auto& p : laserCloudCornerStack)
        p.reset(new pcl::PointCloud<PointType>);
    laserCloudSurfStack.resize(MAX_SLIDE_WINDOW_SIZE);
    for (auto& p : laserCloudSurfStack)
        p.reset(new pcl::PointCloud<PointType>);
    laserCloudNonFeatureStack.resize(MAX_SLIDE_WINDOW_SIZE);
    for (auto& p : laserCloudNonFeatureStack)
        p.reset(new pcl::PointCloud<PointType>);
    laserCloudCornerForMap.reset(new pcl::PointCloud<PointType>);
    laserCloudSurfForMap.reset(new pcl::PointCloud<PointType>);
    laserCloudNonFeatureForMap.reset(new pcl::PointCloud<PointType>);
    transformForMap.setIdentity();
    kdtreeCornerFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
    kdtreeSurfFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
    kdtreeNonFeatureFromLocal.reset(new pcl::KdTreeFLANN<PointType>);

    for (int i = 0; i < localMapWindowSize; i++) {
        localCornerMap[i].reset(new pcl::PointCloud<PointType>);
        localSurfMap[i].reset(new pcl::PointCloud<PointType>);
        localNonFeatureMap[i].reset(new pcl::PointCloud<PointType>);
    }

    downSizeFilterCorner.setLeafSize(filter_corner, filter_corner,
                                     filter_corner);
    downSizeFilterSurf.setLeafSize(filter_surf, filter_surf, filter_surf);
    downSizeFilterNonFeature.setLeafSize(0.4, 0.4, 0.4);
    map_manager.reset(new MAP_MANAGER(filter_corner, filter_surf));
    threadMap = std::thread(&LidarModule::threadMapIncrement, this);

    // Test
    // Scan
    gicpScan2Scan.setCorrespondenceRandomness(10);
    gicpScan2Scan.setMaxCorrespondenceDistance(1.0);
    gicpScan2Scan.setMaximumIterations(32);
    gicpScan2Scan.setTransformationEpsilon(0.01);
    gicpScan2Scan.setEuclideanFitnessEpsilon(0.01);
    gicpScan2Scan.setRANSACIterations(5);
    gicpScan2Scan.setRANSACOutlierRejectionThreshold(1.0);

    pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr tempScan2Scan;
    gicpScan2Scan.setSearchMethodSource(tempScan2Scan, true);
    gicpScan2Scan.setSearchMethodTarget(tempScan2Scan, true);

    // Map
    gicpScan2Map.setCorrespondenceRandomness(20);
    gicpScan2Map.setMaxCorrespondenceDistance(0.5);
    gicpScan2Map.setMaximumIterations(32);
    gicpScan2Map.setTransformationEpsilon(0.01);
    gicpScan2Map.setEuclideanFitnessEpsilon(0.01);
    gicpScan2Map.setRANSACIterations(5);
    gicpScan2Map.setRANSACOutlierRejectionThreshold(1.0);

    pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr tempScan2Map;
    gicpScan2Map.setSearchMethodSource(tempScan2Map, true);
    gicpScan2Map.setSearchMethodTarget(tempScan2Map, true);
}

void LidarModule::slideWindow(const Module::Ptr& prime_module) {
    if (!imu_initialized) {
        if (frame_count == MAX_SLIDE_WINDOW_SIZE) {
            auto slide_f = window_frames.front();
            slide_frames.push_back(slide_f);
            window_frames.pop_front();
            frame_count--;
        }
    } else {
        mergeSlidePointCloud();

        if (!prime_module) {
            while (frame_count >= SLIDE_WINDOW_SIZE) {
                auto slide_f = window_frames.front();
                slide_frames.push_back(slide_f);
                window_frames.pop_front();
                frame_count--;
            }
        } else {
            double front_time = prime_module->window_frames.front()->timeStamp;
            while (window_frames.front()->timeStamp < front_time) {
                auto slide_f = window_frames.front();
                slide_frames.push_back(slide_f);
                window_frames.pop_front();
                frame_count--;
            }
        }
    }
}

void LidarModule::preProcess() {
    int laserCloudCornerFromMapNum =
        map_manager->get_corner_map()->points.size();
    int laserCloudSurfFromMapNum = map_manager->get_surf_map()->points.size();
    int laserCloudCornerFromLocalNum = laserCloudCornerFromLocal->points.size();
    int laserCloudSurfFromLocalNum = laserCloudSurfFromLocal->points.size();

    int stack_count = 0;
    for (const auto& l : window_frames) {
        laserCloudCornerLast[stack_count]->clear();
        laserCloudSurfLast[stack_count]->clear();
        laserCloudNonFeatureLast[stack_count]->clear();

        LidarFrame::Ptr lf = std::dynamic_pointer_cast<LidarFrame>(l);
        for (const auto& p : lf->laserCloud->points) {
            if (std::fabs(p.normal_z - 1.0) < 1e-5)
                laserCloudCornerLast[stack_count]->push_back(p);
            else if (std::fabs(p.normal_z - 2.0) < 1e-5)
                laserCloudSurfLast[stack_count]->push_back(p);
            else if (std::fabs(p.normal_z - 3.0) < 1e-5)
                laserCloudNonFeatureLast[stack_count]->push_back(p);
        }

        laserCloudCornerStack[stack_count]->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast[stack_count]);
        downSizeFilterCorner.filter(*laserCloudCornerStack[stack_count]);

        laserCloudSurfStack[stack_count]->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast[stack_count]);
        downSizeFilterSurf.filter(*laserCloudSurfStack[stack_count]);

        laserCloudNonFeatureStack[stack_count]->clear();
        downSizeFilterNonFeature.setInputCloud(
            laserCloudNonFeatureLast[stack_count]);
        downSizeFilterNonFeature.filter(
            *laserCloudNonFeatureStack[stack_count]);
        stack_count++;
    }

    if (((laserCloudCornerFromMapNum > 0 && laserCloudSurfFromMapNum > 100) ||
         (laserCloudCornerFromLocalNum > 0 &&
          laserCloudSurfFromLocalNum > 100))) {
        kdtreeCornerFromLocal->setInputCloud(laserCloudCornerFromLocal);
        kdtreeSurfFromLocal->setInputCloud(laserCloudSurfFromLocal);
        kdtreeNonFeatureFromLocal->setInputCloud(laserCloudNonFeatureFromLocal);

        std::unique_lock<std::mutex> locker3(map_manager->mtx_MapManager);

        for (int i = 0; i < 4851; i++) {
            CornerKdMap[i] = map_manager->getCornerKdMap(i);
            SurfKdMap[i] = map_manager->getSurfKdMap(i);
            NonFeatureKdMap[i] = map_manager->getNonFeatureKdMap(i);

            GlobalSurfMap[i] = map_manager->laserCloudSurf_for_match[i];
            GlobalCornerMap[i] = map_manager->laserCloudCorner_for_match[i];
            GlobalNonFeatureMap[i] =
                map_manager->laserCloudNonFeature_for_match[i];
        }
        laserCenWidth_last = map_manager->get_laserCloudCenWidth_last();
        laserCenHeight_last = map_manager->get_laserCloudCenHeight_last();
        laserCenDepth_last = map_manager->get_laserCloudCenDepth_last();

        locker3.unlock();

        int windowSize = window_frames.size();
        vLineFeatures.resize(windowSize);
        for (auto& v : vLineFeatures) {
            v.reserve(2000);
        }

        vPlanFeatures.resize(windowSize);
        for (auto& v : vPlanFeatures) {
            v.reserve(2000);
        }

        vNonFeatures.resize(windowSize);
        for (auto& v : vNonFeatures) {
            v.reserve(2000);
        }

        plan_weight_tan = 0.0003;
        thres_dist = 1.0;

        to_be_used = true;
    }
}

void LidarModule::mergeSlidePointCloud() {
    pcl::PointCloud<PointType>::Ptr slide_pointcloud_corner(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr slide_pointcloud_surf(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr slide_pointcloud_nonfeature(
        new pcl::PointCloud<PointType>);

    auto curr_frame = window_frames.begin();
    std::advance(curr_frame, slide_window_size - 1);

    auto curr_sensor_id = (*curr_frame)->sensor_id;
    Eigen::Matrix4d currTbl = ex_pose[curr_sensor_id];
    Eigen::Matrix4d currTwl = Eigen::Matrix4d::Identity();
    currTwl.topLeftCorner(3, 3) = (*curr_frame)->Q * currTbl.block<3, 3>(0, 0);
    currTwl.topRightCorner(3, 1) =
        (*curr_frame)->Q * currTbl.block<3, 1>(0, 3) + (*curr_frame)->P;

    for (int i = 0; i < slide_window_size - 1; i++) {
        auto prev_frame = window_frames.begin();
        std::advance(prev_frame, i);

        auto prev_sensor_id = (*prev_frame)->sensor_id;
        Eigen::Matrix4d prevTbl = ex_pose[prev_sensor_id];
        Eigen::Matrix4d prevTwl = Eigen::Matrix4d::Identity();
        prevTwl.topLeftCorner(3, 3) =
            (*prev_frame)->Q * prevTbl.block<3, 3>(0, 0);
        prevTwl.topRightCorner(3, 1) =
            (*prev_frame)->Q * prevTbl.block<3, 1>(0, 3) + (*prev_frame)->P;

        Eigen::Matrix4d Tcp = currTwl.inverse() * prevTwl;
        for (int j = 0; j < laserCloudCornerStack[i]->size(); j++) {
            PointType pointSel;
            MAP_MANAGER::pointAssociateToMap(
                &laserCloudCornerStack[i]->points[j], &pointSel, Tcp);
            slide_pointcloud_corner->push_back(pointSel);
        }

        for (int j = 0; j < laserCloudSurfStack[i]->size(); j++) {
            PointType pointSel;
            MAP_MANAGER::pointAssociateToMap(&laserCloudSurfStack[i]->points[j],
                                             &pointSel, Tcp);
            slide_pointcloud_surf->push_back(pointSel);
        }

        for (int j = 0; j < laserCloudNonFeatureStack[i]->size(); j++) {
            PointType pointSel;
            MAP_MANAGER::pointAssociateToMap(
                &laserCloudNonFeatureStack[i]->points[j], &pointSel, Tcp);
            slide_pointcloud_nonfeature->push_back(pointSel);
        }
    }

    std::unique_lock<std::mutex> locker(mtx_Map);
    laserCloudCornerForMap = slide_pointcloud_corner;
    laserCloudSurfForMap = slide_pointcloud_surf;
    laserCloudNonFeatureForMap = slide_pointcloud_nonfeature;
    transformForMap = currTwl;
    MapIncrementLocal(laserCloudCornerForMap, laserCloudSurfForMap,
                      laserCloudNonFeatureForMap, transformForMap);
    locker.unlock();
}

void LidarModule::postProcess() {
    if (!to_be_used)
        return;
}

void LidarModule::addResidualBlock(int iterOpt) {
    if (!to_be_used)
        return;

    std::string lidar_id = window_frames.back()->sensor_id;
    q_before_opti = window_frames.back()->Q;
    t_before_opti = window_frames.back()->P;

    Eigen::Matrix<double, 3, 3> exRbl = ex_pose[lidar_id].block<3, 3>(0, 0);
    Eigen::Matrix<double, 3, 1> exPbl = ex_pose[lidar_id].block<3, 1>(0, 3);

    int windowSize = window_frames.size();
    std::vector<std::vector<ceres::CostFunction*>> edgesLine(windowSize);
    std::vector<std::vector<ceres::CostFunction*>> edgesPlan(windowSize);
    std::vector<std::vector<ceres::CostFunction*>> edgesNon(windowSize);
    for (int f = 0; f < windowSize; ++f) {
        auto frame_curr = window_frames.begin();
        std::advance(frame_curr, f);
        Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
        transformTobeMapped.topLeftCorner(3, 3) = (*frame_curr)->Q * exRbl;
        transformTobeMapped.topRightCorner(3, 1) =
            (*frame_curr)->Q * exPbl + (*frame_curr)->P;

        std::thread threads[3];
        threads[0] = std::thread(
            &LidarModule::processPointToLine, this, std::ref(edgesLine[f]),
            std::ref(vLineFeatures[f]), std::ref(laserCloudCornerStack[f]),
            std::ref(laserCloudCornerFromLocal),
            std::ref(kdtreeCornerFromLocal), std::ref(transformTobeMapped));

        threads[1] = std::thread(
            &LidarModule::processPointToPlan, this, std::ref(edgesPlan[f]),
            std::ref(vPlanFeatures[f]), std::ref(laserCloudSurfStack[f]),
            std::ref(laserCloudSurfFromLocal), std::ref(kdtreeSurfFromLocal),
            std::ref(transformTobeMapped));

        threads[2] = std::thread(
            &LidarModule::processNonFeatureICP, this, std::ref(edgesNon[f]),
            std::ref(vNonFeatures[f]), std::ref(laserCloudNonFeatureStack[f]),
            std::ref(laserCloudNonFeatureFromLocal),
            std::ref(kdtreeNonFeatureFromLocal), std::ref(transformTobeMapped));

        threads[0].join();
        threads[1].join();
        threads[2].join();
    }

    {
        residual_block_ids.clear();
        residual_block_ids["lidar_corner"] =
            std::vector<ceres::ResidualBlockId>();
        residual_block_ids["lidar_surf"] =
            std::vector<ceres::ResidualBlockId>();
        residual_block_ids["lidar_nonfeat"] =
            std::vector<ceres::ResidualBlockId>();
    }

    // create huber loss function
    ceres::LossFunction* loss_function = NULL;
    loss_function = NULL;

    int cntSurf = 0;
    int cntCorner = 0;
    int cntNon = 0;
    thres_dist = 1.0;
    if (iterOpt == 0) {
        for (int f = 0; f < windowSize; ++f) {
            int cntFtu = 0;
            for (auto& e : edgesLine[f]) {
                if (std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5) {
                    auto re_id = problem->AddResidualBlock(
                        e, loss_function, para_Pose[f], para_Ex_Pose[lidar_id]);
                    residual_block_ids["lidar_corner"].push_back(re_id);
                    vLineFeatures[f][cntFtu].valid = true;
                } else {
                    vLineFeatures[f][cntFtu].valid = false;
                }
                cntFtu++;
                cntCorner++;
            }

            cntFtu = 0;
            for (auto& e : edgesPlan[f]) {
                if (std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5) {
                    auto re_id = problem->AddResidualBlock(
                        e, loss_function, para_Pose[f], para_Ex_Pose[lidar_id]);
                    residual_block_ids["lidar_surf"].push_back(re_id);
                    vPlanFeatures[f][cntFtu].valid = true;
                } else {
                    vPlanFeatures[f][cntFtu].valid = false;
                }
                cntFtu++;
                cntSurf++;
            }

            cntFtu = 0;
            for (auto& e : edgesNon[f]) {
                if (std::fabs(vNonFeatures[f][cntFtu].error) > 1e-5) {
                    auto re_id = problem->AddResidualBlock(
                        e, loss_function, para_Pose[f], para_Ex_Pose[lidar_id]);
                    residual_block_ids["lidar_nonfeat"].push_back(re_id);
                    vNonFeatures[f][cntFtu].valid = true;
                } else {
                    vNonFeatures[f][cntFtu].valid = false;
                }
                cntFtu++;
                cntNon++;
            }
        }
    } else {
        for (int f = 0; f < windowSize; ++f) {
            int cntFtu = 0;
            for (auto& e : edgesLine[f]) {
                if (vLineFeatures[f][cntFtu].valid) {
                    auto re_id = problem->AddResidualBlock(
                        e, loss_function, para_Pose[f], para_Ex_Pose[lidar_id]);
                    residual_block_ids["lidar_corner"].push_back(re_id);
                }
                cntFtu++;
                cntCorner++;
            }

            cntFtu = 0;
            for (auto& e : edgesPlan[f]) {
                if (vPlanFeatures[f][cntFtu].valid) {
                    auto re_id = problem->AddResidualBlock(
                        e, loss_function, para_Pose[f], para_Ex_Pose[lidar_id]);
                    residual_block_ids["lidar_surf"].push_back(re_id);
                }
                cntFtu++;
                cntSurf++;
            }

            cntFtu = 0;
            for (auto& e : edgesNon[f]) {
                if (vNonFeatures[f][cntFtu].valid) {
                    auto re_id = problem->AddResidualBlock(
                        e, loss_function, para_Pose[f], para_Ex_Pose[lidar_id]);
                    residual_block_ids["lidar_nonfeat"].push_back(re_id);
                }
                cntFtu++;
                cntNon++;
            }
        }
    }

    std::cout << "cntSurf : " << cntSurf << std::endl;
    std::cout << "cntCorner : " << cntCorner << std::endl;
    std::cout << "cntNon : " << cntNon << std::endl;
}

void LidarModule::vector2double() {
    for (int i = 0; i < window_frames.size(); i++) {
        auto lf = window_frames.begin();
        std::advance(lf, i);
        Eigen::Map<Eigen::Matrix<double, 6, 1>> PR(para_Pose[i]);
        PR.segment<3>(0) = (*lf)->P;
        PR.segment<3>(3) = Sophus::SO3d((*lf)->Q).log();

        Eigen::Map<Eigen::Matrix<double, 9, 1>> VBias(para_SpeedBias[i]);
        VBias.segment<3>(0) = (*lf)->V;
        VBias.segment<3>(3) = (*lf)->bg;
        VBias.segment<3>(6) = (*lf)->ba;
    }

    for (auto l : ex_pose) {
        std::cout << para_Ex_Pose.size() << std::endl;
        if (para_Ex_Pose.find(l.first) == para_Ex_Pose.end())
            para_Ex_Pose[l.first] = new double[SIZE_POSE];
        Eigen::Map<Eigen::Matrix<double, 6, 1>> Exbl(para_Ex_Pose[l.first]);
        const auto& exTbl = l.second;
        Exbl.segment<3>(0) = exTbl.block<3, 1>(0, 3);
        Exbl.segment<3>(3) =
            Sophus::SO3d(Eigen::Quaterniond(exTbl.block<3, 3>(0, 0))
                             .normalized()
                             .toRotationMatrix())
                .log();
    }
}

void LidarModule::addParameter() {
    if (!to_be_used)
        return;

    for (int i = 0; i < window_frames.size(); i++) {
        ceres::LocalParameterization* local_parameterization = NULL;
        problem->AddParameterBlock(para_Pose[i], SIZE_POSE,
                                   local_parameterization);
        problem->AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    for (auto c : para_Ex_Pose) {
        ceres::LocalParameterization* local_parameterization = NULL;
        problem->AddParameterBlock(c.second, SIZE_POSE, local_parameterization);
        if (1) {
            problem->SetParameterBlockConstant(c.second);
        }
    }
}

void LidarModule::double2vector() {
    for (auto l : ex_pose) {
        Eigen::Map<Eigen::Matrix<double, 6, 1>> Exbl(para_Ex_Pose[l.first]);
        l.second = Eigen::Matrix4d::Identity();
        l.second.block<3, 3>(0, 0) =
            (Sophus::SO3d::exp(Exbl.segment<3>(3)).unit_quaternion())
                .toRotationMatrix();
        l.second.block<3, 1>(0, 3) = Exbl.segment<3>(0);
    }

    for (int i = 0; i < window_frames.size(); i++) {
        auto lf = window_frames.begin();
        std::advance(lf, i);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> PR(para_Pose[i]);
        Eigen::Map<const Eigen::Matrix<double, 9, 1>> VBias(para_SpeedBias[i]);
        (*lf)->P = PR.segment<3>(0);
        (*lf)->Q = Sophus::SO3d::exp(PR.segment<3>(3)).unit_quaternion();
        (*lf)->V = VBias.segment<3>(0);
        (*lf)->bg = VBias.segment<3>(3);
        (*lf)->ba = VBias.segment<3>(6);
        (*lf)->ExT_ = ex_pose[(*lf)->sensor_id];
    }
}

void LidarModule::marginalization1(
    MarginalizationInfo* last_marginalization_info,
    std::vector<double*>& last_marginalization_parameter_blocks,
    MarginalizationInfo* marginalization_info, int slide_win_size) {
    if (!to_be_used)
        return;

    std::vector<std::vector<ceres::CostFunction*>> edgesLine(slide_win_size);
    std::vector<std::vector<ceres::CostFunction*>> edgesPlan(slide_win_size);
    std::vector<std::vector<ceres::CostFunction*>> edgesNon(slide_win_size);

    for (size_t f = 0; f < slide_win_size; f++) {
        auto curr_frame = window_frames.begin();
        std::advance(curr_frame, f);

        std::string lidar_id = (*curr_frame)->sensor_id;
        Eigen::Matrix<double, 3, 3> exRbl = ex_pose[lidar_id].block<3, 3>(0, 0);
        Eigen::Matrix<double, 3, 1> exPbl = ex_pose[lidar_id].block<3, 1>(0, 3);
        Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
        transformTobeMapped.topLeftCorner(3, 3) = (*curr_frame)->Q * exRbl;
        transformTobeMapped.topRightCorner(3, 1) =
            (*curr_frame)->Q * exPbl + (*curr_frame)->P;

        std::thread threads[3];
        threads[0] = std::thread(
            &LidarModule::processPointToLine, this, std::ref(edgesLine[f]),
            std::ref(vLineFeatures[f]), std::ref(laserCloudCornerStack[f]),
            std::ref(laserCloudCornerFromLocal),
            std::ref(kdtreeCornerFromLocal), std::ref(transformTobeMapped));

        threads[1] = std::thread(
            &LidarModule::processPointToPlan, this, std::ref(edgesPlan[f]),
            std::ref(vPlanFeatures[f]), std::ref(laserCloudSurfStack[f]),
            std::ref(laserCloudSurfFromLocal), std::ref(kdtreeSurfFromLocal),
            std::ref(transformTobeMapped));

        threads[2] = std::thread(
            &LidarModule::processNonFeatureICP, this, std::ref(edgesNon[f]),
            std::ref(vNonFeatures[f]), std::ref(laserCloudNonFeatureStack[f]),
            std::ref(laserCloudNonFeatureFromLocal),
            std::ref(kdtreeNonFeatureFromLocal), std::ref(transformTobeMapped));

        threads[0].join();
        threads[1].join();
        threads[2].join();
        int cntFtu = 0;
        for (auto& e : edgesLine[f]) {
            if (vLineFeatures[f][cntFtu].valid) {
                auto* residual_block_info = new ResidualBlockInfo(
                    e, nullptr,
                    std::vector<double*>{para_Pose[f], para_Ex_Pose[lidar_id]},
                    std::vector<int>{0});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            cntFtu++;
        }

        cntFtu = 0;
        for (auto& e : edgesPlan[f]) {
            if (vPlanFeatures[f][cntFtu].valid) {
                auto* residual_block_info = new ResidualBlockInfo(
                    e, nullptr,
                    std::vector<double*>{para_Pose[f], para_Ex_Pose[lidar_id]},
                    std::vector<int>{0});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            cntFtu++;
        }

        cntFtu = 0;
        for (auto& e : edgesNon[f]) {
            if (vNonFeatures[f][cntFtu].valid) {
                auto* residual_block_info = new ResidualBlockInfo(
                    e, nullptr,
                    std::vector<double*>{para_Pose[f], para_Ex_Pose[lidar_id]},
                    std::vector<int>{0});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            cntFtu++;
        }
    }
}

void LidarModule::marginalization2(
    std::unordered_map<long, double*>& addr_shift, int slide_win_size) {
    if (!to_be_used)
        return;

    for (int i = slide_win_size; i < window_frames.size(); i++) {
        addr_shift[reinterpret_cast<long>(para_Pose[i])] =
            para_Pose[i - slide_win_size];
        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] =
            para_SpeedBias[i - slide_win_size];
    }
    for (auto c : para_Ex_Pose)
        addr_shift[reinterpret_cast<long>(c.second)] = c.second;
}

bool LidarModule::getFineSolveFlag() {
    if (!to_be_used)
        return false;

    auto q_after_opti = window_frames.back()->Q;
    auto t_after_opti = window_frames.back()->P;
    double deltaR =
        (q_before_opti.angularDistance(q_after_opti)) * 180.0 / M_PI;
    double deltaT = (t_before_opti - t_after_opti).norm();

    if (deltaR < 0.05 && deltaT < 0.05)
        return true;
    return false;
}

[[noreturn]] void LidarModule::threadMapIncrement() {
    pcl::PointCloud<PointType>::Ptr laserCloudCorner(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr laserCloudSurf(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr laserCloudNonFeature(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr laserCloudCorner_to_map(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr laserCloudSurf_to_map(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr laserCloudNonFeature_to_map(
        new pcl::PointCloud<PointType>);
    Eigen::Matrix4d transform;
    while (true) {
        std::unique_lock<std::mutex> locker(mtx_Map);
        if (!laserCloudSurfForMap->empty()) {
            map_update_ID++;

            map_manager->featureAssociateToMap(
                laserCloudCornerForMap, laserCloudSurfForMap,
                laserCloudNonFeatureForMap, laserCloudCorner, laserCloudSurf,
                laserCloudNonFeature, transformForMap);
            laserCloudCornerForMap->clear();
            laserCloudSurfForMap->clear();
            laserCloudNonFeatureForMap->clear();
            transform = transformForMap;

            locker.unlock();

            *laserCloudCorner_to_map += *laserCloudCorner;
            *laserCloudSurf_to_map += *laserCloudSurf;
            *laserCloudNonFeature_to_map += *laserCloudNonFeature;

            laserCloudCorner->clear();
            laserCloudSurf->clear();
            laserCloudNonFeature->clear();

            if (map_update_ID % map_skip_frame == 0) {
                map_manager->MapIncrement(
                    laserCloudCorner_to_map, laserCloudSurf_to_map,
                    laserCloudNonFeature_to_map, transform);

                laserCloudCorner_to_map->clear();
                laserCloudSurf_to_map->clear();
                laserCloudNonFeature_to_map->clear();
            }

        } else
            locker.unlock();

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void LidarModule::processPointToLine(
    std::vector<ceres::CostFunction*>& edges,
    std::vector<FeatureLine>& vLineFeatures,
    const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
    const pcl::PointCloud<PointType>::Ptr& laserCloudCornerLocal,
    const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
    const Eigen::Matrix4d& m4d) {
    if (!vLineFeatures.empty()) {
        for (const auto& l : vLineFeatures) {
            auto* e = Cost_NavState_IMU_Line::Create(
                l.pointOri, l.lineP1, l.lineP2,
                Eigen::Matrix<double, 1, 1>(1 / IMUIntegrator::lidar_m));
            edges.push_back(e);
        }
        return;
    }
    PointType _pointOri, _pointSel, _coeff;
    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;
    std::vector<int> _pointSearchInd2;
    std::vector<float> _pointSearchSqDis2;

    Eigen::Matrix<double, 3, 3> _matA1;
    _matA1.setZero();

    int laserCloudCornerStackNum = laserCloudCorner->points.size();
    pcl::PointCloud<PointType>::Ptr kd_pointcloud(
        new pcl::PointCloud<PointType>);
    int debug_num1 = 0;
    int debug_num2 = 0;
    int debug_num12 = 0;
    int debug_num22 = 0;
    for (int i = 0; i < laserCloudCornerStackNum; i++) {
        _pointOri = laserCloudCorner->points[i];
        MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
        int id = map_manager->FindUsedCornerMap(&_pointSel, laserCenWidth_last,
                                                laserCenHeight_last,
                                                laserCenDepth_last);

        if (id == 5000)
            continue;

        if (std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||
            std::isnan(_pointSel.z))
            continue;

        if (GlobalCornerMap[id].points.size() > 100) {
            CornerKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd,
                                           _pointSearchSqDis);

            if (_pointSearchSqDis[4] < thres_dist) {
                debug_num1++;
                float cx = 0;
                float cy = 0;
                float cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += GlobalCornerMap[id].points[_pointSearchInd[j]].x;
                    cy += GlobalCornerMap[id].points[_pointSearchInd[j]].y;
                    cz += GlobalCornerMap[id].points[_pointSearchInd[j]].z;
                }
                cx /= 5;
                cy /= 5;
                cz /= 5;

                float a11 = 0;
                float a12 = 0;
                float a13 = 0;
                float a22 = 0;
                float a23 = 0;
                float a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax =
                        GlobalCornerMap[id].points[_pointSearchInd[j]].x - cx;
                    float ay =
                        GlobalCornerMap[id].points[_pointSearchInd[j]].y - cy;
                    float az =
                        GlobalCornerMap[id].points[_pointSearchInd[j]].z - cz;

                    a11 += ax * ax;
                    a12 += ax * ay;
                    a13 += ax * az;
                    a22 += ay * ay;
                    a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

                _matA1(0, 0) = a11;
                _matA1(0, 1) = a12;
                _matA1(0, 2) = a13;
                _matA1(1, 0) = a12;
                _matA1(1, 1) = a22;
                _matA1(1, 2) = a23;
                _matA1(2, 0) = a13;
                _matA1(2, 1) = a23;
                _matA1(2, 2) = a33;

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
                Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

                if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
                    debug_num12++;
                    float x1 = cx + 0.1 * unit_direction[0];
                    float y1 = cy + 0.1 * unit_direction[1];
                    float z1 = cz + 0.1 * unit_direction[2];
                    float x2 = cx - 0.1 * unit_direction[0];
                    float y2 = cy - 0.1 * unit_direction[1];
                    float z2 = cz - 0.1 * unit_direction[2];

                    Eigen::Vector3d tripod1(x1, y1, z1);
                    Eigen::Vector3d tripod2(x2, y2, z2);
                    auto* e = Cost_NavState_IMU_Line::Create(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        tripod1, tripod2,
                        Eigen::Matrix<double, 1, 1>(1 /
                                                    IMUIntegrator::lidar_m));
                    edges.push_back(e);
                    vLineFeatures.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        tripod1, tripod2);
                    vLineFeatures.back().ComputeError(m4d);

                    continue;
                }
            }
        }

        if (laserCloudCornerLocal->points.size() > 20) {
            kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2,
                                        _pointSearchSqDis2);
            if (_pointSearchSqDis2[4] < thres_dist) {
                debug_num2++;
                float cx = 0;
                float cy = 0;
                float cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerLocal->points[_pointSearchInd2[j]].x;
                    cy += laserCloudCornerLocal->points[_pointSearchInd2[j]].y;
                    cz += laserCloudCornerLocal->points[_pointSearchInd2[j]].z;
                }
                cx /= 5;
                cy /= 5;
                cz /= 5;

                float a11 = 0;
                float a12 = 0;
                float a13 = 0;
                float a22 = 0;
                float a23 = 0;
                float a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax =
                        laserCloudCornerLocal->points[_pointSearchInd2[j]].x -
                        cx;
                    float ay =
                        laserCloudCornerLocal->points[_pointSearchInd2[j]].y -
                        cy;
                    float az =
                        laserCloudCornerLocal->points[_pointSearchInd2[j]].z -
                        cz;

                    a11 += ax * ax;
                    a12 += ax * ay;
                    a13 += ax * az;
                    a22 += ay * ay;
                    a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

                _matA1(0, 0) = a11;
                _matA1(0, 1) = a12;
                _matA1(0, 2) = a13;
                _matA1(1, 0) = a12;
                _matA1(1, 1) = a22;
                _matA1(1, 2) = a23;
                _matA1(2, 0) = a13;
                _matA1(2, 1) = a23;
                _matA1(2, 2) = a33;

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
                Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

                if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
                    debug_num22++;
                    float x1 = cx + 0.1 * unit_direction[0];
                    float y1 = cy + 0.1 * unit_direction[1];
                    float z1 = cz + 0.1 * unit_direction[2];
                    float x2 = cx - 0.1 * unit_direction[0];
                    float y2 = cy - 0.1 * unit_direction[1];
                    float z2 = cz - 0.1 * unit_direction[2];

                    Eigen::Vector3d tripod1(x1, y1, z1);
                    Eigen::Vector3d tripod2(x2, y2, z2);
                    auto* e = Cost_NavState_IMU_Line::Create(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        tripod1, tripod2,
                        Eigen::Matrix<double, 1, 1>(1 /
                                                    IMUIntegrator::lidar_m));
                    edges.push_back(e);
                    vLineFeatures.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        tripod1, tripod2);
                    vLineFeatures.back().ComputeError(m4d);
                }
            }
        }
    }
}

void LidarModule::processPointToPlan(
    std::vector<ceres::CostFunction*>& edges,
    std::vector<FeaturePlan>& vPlanFeatures,
    const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
    const pcl::PointCloud<PointType>::Ptr& laserCloudSurfLocal,
    const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
    const Eigen::Matrix4d& m4d) {
    if (!vPlanFeatures.empty()) {
        for (const auto& p : vPlanFeatures) {
            auto* e = Cost_NavState_IMU_Plan::Create(
                p.pointOri, p.pa, p.pb, p.pc, p.pd,
                Eigen::Matrix<double, 1, 1>(1 / IMUIntegrator::lidar_m));
            edges.push_back(e);
        }
        return;
    }
    PointType _pointOri, _pointSel, _coeff;
    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;
    std::vector<int> _pointSearchInd2;
    std::vector<float> _pointSearchSqDis2;

    Eigen::Matrix<double, 5, 3> _matA0;
    _matA0.setZero();
    Eigen::Matrix<double, 5, 1> _matB0;
    _matB0.setOnes();
    _matB0 *= -1;
    Eigen::Matrix<double, 3, 1> _matX0;
    _matX0.setZero();
    int laserCloudSurfStackNum = laserCloudSurf->points.size();

    int debug_num1 = 0;
    int debug_num2 = 0;
    int debug_num12 = 0;
    int debug_num22 = 0;
    for (int i = 0; i < laserCloudSurfStackNum; i++) {
        _pointOri = laserCloudSurf->points[i];
        MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);

        int id = map_manager->FindUsedSurfMap(&_pointSel, laserCenWidth_last,
                                              laserCenHeight_last,
                                              laserCenDepth_last);

        if (id == 5000)
            continue;

        if (std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||
            std::isnan(_pointSel.z))
            continue;

        if (GlobalSurfMap[id].points.size() > 50) {
            SurfKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd,
                                         _pointSearchSqDis);

            if (_pointSearchSqDis[4] < 1.0) {
                debug_num1++;
                for (int j = 0; j < 5; j++) {
                    _matA0(j, 0) =
                        GlobalSurfMap[id].points[_pointSearchInd[j]].x;
                    _matA0(j, 1) =
                        GlobalSurfMap[id].points[_pointSearchInd[j]].y;
                    _matA0(j, 2) =
                        GlobalSurfMap[id].points[_pointSearchInd[j]].z;
                }
                _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

                float pa = _matX0(0, 0);
                float pb = _matX0(1, 0);
                float pc = _matX0(2, 0);
                float pd = 1;

                float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (std::fabs(
                            pa *
                                GlobalSurfMap[id].points[_pointSearchInd[j]].x +
                            pb *
                                GlobalSurfMap[id].points[_pointSearchInd[j]].y +
                            pc *
                                GlobalSurfMap[id].points[_pointSearchInd[j]].z +
                            pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    debug_num12++;
                    auto* e = Cost_NavState_IMU_Plan::Create(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd,
                        Eigen::Matrix<double, 1, 1>(1 /
                                                    IMUIntegrator::lidar_m));
                    edges.push_back(e);
                    vPlanFeatures.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd);
                    vPlanFeatures.back().ComputeError(m4d);

                    continue;
                }
            }
        }
        if (laserCloudSurfLocal->points.size() > 20) {
            kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2,
                                        _pointSearchSqDis2);
            if (_pointSearchSqDis2[4] < 1.0) {
                debug_num2++;
                for (int j = 0; j < 5; j++) {
                    _matA0(j, 0) =
                        laserCloudSurfLocal->points[_pointSearchInd2[j]].x;
                    _matA0(j, 1) =
                        laserCloudSurfLocal->points[_pointSearchInd2[j]].y;
                    _matA0(j, 2) =
                        laserCloudSurfLocal->points[_pointSearchInd2[j]].z;
                }
                _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

                float pa = _matX0(0, 0);
                float pb = _matX0(1, 0);
                float pc = _matX0(2, 0);
                float pd = 1;

                float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (std::fabs(pa * laserCloudSurfLocal
                                           ->points[_pointSearchInd2[j]]
                                           .x +
                                  pb * laserCloudSurfLocal
                                           ->points[_pointSearchInd2[j]]
                                           .y +
                                  pc * laserCloudSurfLocal
                                           ->points[_pointSearchInd2[j]]
                                           .z +
                                  pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    debug_num22++;
                    auto* e = Cost_NavState_IMU_Plan::Create(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd,
                        Eigen::Matrix<double, 1, 1>(1 /
                                                    IMUIntegrator::lidar_m));
                    edges.push_back(e);
                    vPlanFeatures.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd);
                    vPlanFeatures.back().ComputeError(m4d);
                }
            }
        }
    }
}

void LidarModule::processPointToPlanVec(
    std::vector<ceres::CostFunction*>& edges,
    std::vector<FeaturePlanVec>& vPlanFeatures,
    const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
    const pcl::PointCloud<PointType>::Ptr& laserCloudSurfLocal,
    const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
    const Eigen::Matrix4d& m4d) {
    if (!vPlanFeatures.empty()) {
        for (const auto& p : vPlanFeatures) {
            auto* e = Cost_NavState_IMU_Plan_Vec::Create(
                p.pointOri, p.pointProj, p.sqrt_info);
            edges.push_back(e);
        }
        return;
    }
    PointType _pointOri, _pointSel, _coeff;
    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;
    std::vector<int> _pointSearchInd2;
    std::vector<float> _pointSearchSqDis2;

    Eigen::Matrix<double, 5, 3> _matA0;
    _matA0.setZero();
    Eigen::Matrix<double, 5, 1> _matB0;
    _matB0.setOnes();
    _matB0 *= -1;
    Eigen::Matrix<double, 3, 1> _matX0;
    _matX0.setZero();
    int laserCloudSurfStackNum = laserCloudSurf->points.size();

    int debug_num1 = 0;
    int debug_num2 = 0;
    int debug_num12 = 0;
    int debug_num22 = 0;
    for (int i = 0; i < laserCloudSurfStackNum; i++) {
        _pointOri = laserCloudSurf->points[i];
        MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);

        int id = map_manager->FindUsedSurfMap(&_pointSel, laserCenWidth_last,
                                              laserCenHeight_last,
                                              laserCenDepth_last);

        if (id == 5000)
            continue;

        if (std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||
            std::isnan(_pointSel.z))
            continue;

        if (GlobalSurfMap[id].points.size() > 50) {
            SurfKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd,
                                         _pointSearchSqDis);

            if (_pointSearchSqDis[4] < thres_dist) {
                debug_num1++;
                for (int j = 0; j < 5; j++) {
                    _matA0(j, 0) =
                        GlobalSurfMap[id].points[_pointSearchInd[j]].x;
                    _matA0(j, 1) =
                        GlobalSurfMap[id].points[_pointSearchInd[j]].y;
                    _matA0(j, 2) =
                        GlobalSurfMap[id].points[_pointSearchInd[j]].z;
                }
                _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

                float pa = _matX0(0, 0);
                float pb = _matX0(1, 0);
                float pc = _matX0(2, 0);
                float pd = 1;

                float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (std::fabs(
                            pa *
                                GlobalSurfMap[id].points[_pointSearchInd[j]].x +
                            pb *
                                GlobalSurfMap[id].points[_pointSearchInd[j]].y +
                            pc *
                                GlobalSurfMap[id].points[_pointSearchInd[j]].z +
                            pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    debug_num12++;
                    double dist = pa * _pointSel.x + pb * _pointSel.y +
                                  pc * _pointSel.z + pd;
                    Eigen::Vector3d omega(pa, pb, pc);
                    Eigen::Vector3d point_proj =
                        Eigen::Vector3d(_pointSel.x, _pointSel.y, _pointSel.z) -
                        (dist * omega);
                    Eigen::Vector3d e1(1, 0, 0);
                    Eigen::Matrix3d J = e1 * omega.transpose();
                    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                        J, Eigen::ComputeThinU | Eigen::ComputeThinV);
                    Eigen::Matrix3d R_svd =
                        svd.matrixV() * svd.matrixU().transpose();
                    Eigen::Matrix3d info = (1.0 / IMUIntegrator::lidar_m) *
                                           Eigen::Matrix3d::Identity();
                    info(1, 1) *= plan_weight_tan;
                    info(2, 2) *= plan_weight_tan;
                    Eigen::Matrix3d sqrt_info = info * R_svd.transpose();

                    auto* e = Cost_NavState_IMU_Plan_Vec::Create(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        point_proj, sqrt_info);
                    edges.push_back(e);
                    vPlanFeatures.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        point_proj, sqrt_info);
                    vPlanFeatures.back().ComputeError(m4d);

                    continue;
                }
            }
        }

        if (laserCloudSurfLocal->points.size() > 20) {
            kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2,
                                        _pointSearchSqDis2);
            if (_pointSearchSqDis2[4] < thres_dist) {
                debug_num2++;
                for (int j = 0; j < 5; j++) {
                    _matA0(j, 0) =
                        laserCloudSurfLocal->points[_pointSearchInd2[j]].x;
                    _matA0(j, 1) =
                        laserCloudSurfLocal->points[_pointSearchInd2[j]].y;
                    _matA0(j, 2) =
                        laserCloudSurfLocal->points[_pointSearchInd2[j]].z;
                }
                _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

                float pa = _matX0(0, 0);
                float pb = _matX0(1, 0);
                float pc = _matX0(2, 0);
                float pd = 1;

                float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (std::fabs(pa * laserCloudSurfLocal
                                           ->points[_pointSearchInd2[j]]
                                           .x +
                                  pb * laserCloudSurfLocal
                                           ->points[_pointSearchInd2[j]]
                                           .y +
                                  pc * laserCloudSurfLocal
                                           ->points[_pointSearchInd2[j]]
                                           .z +
                                  pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    debug_num22++;
                    double dist = pa * _pointSel.x + pb * _pointSel.y +
                                  pc * _pointSel.z + pd;
                    Eigen::Vector3d omega(pa, pb, pc);
                    Eigen::Vector3d point_proj =
                        Eigen::Vector3d(_pointSel.x, _pointSel.y, _pointSel.z) -
                        (dist * omega);
                    Eigen::Vector3d e1(1, 0, 0);
                    Eigen::Matrix3d J = e1 * omega.transpose();
                    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                        J, Eigen::ComputeThinU | Eigen::ComputeThinV);
                    Eigen::Matrix3d R_svd =
                        svd.matrixV() * svd.matrixU().transpose();
                    Eigen::Matrix3d info = (1.0 / IMUIntegrator::lidar_m) *
                                           Eigen::Matrix3d::Identity();
                    info(1, 1) *= plan_weight_tan;
                    info(2, 2) *= plan_weight_tan;
                    Eigen::Matrix3d sqrt_info = info * R_svd.transpose();

                    auto* e = Cost_NavState_IMU_Plan_Vec::Create(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        point_proj, sqrt_info);
                    edges.push_back(e);
                    vPlanFeatures.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        point_proj, sqrt_info);
                    vPlanFeatures.back().ComputeError(m4d);
                }
            }
        }
    }
}

void LidarModule::processNonFeatureICP(
    std::vector<ceres::CostFunction*>& edges,
    std::vector<FeatureNon>& vNonFeatures,
    const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
    const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureLocal,
    const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
    const Eigen::Matrix4d& m4d) {
    if (!vNonFeatures.empty()) {
        for (const auto& p : vNonFeatures) {
            auto* e = Cost_NonFeature_ICP::Create(
                p.pointOri, p.pa, p.pb, p.pc, p.pd,
                Eigen::Matrix<double, 1, 1>(1 / IMUIntegrator::lidar_m));
            edges.push_back(e);
        }
        return;
    }

    PointType _pointOri, _pointSel, _coeff;
    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;
    std::vector<int> _pointSearchInd2;
    std::vector<float> _pointSearchSqDis2;

    Eigen::Matrix<double, 5, 3> _matA0;
    _matA0.setZero();
    Eigen::Matrix<double, 5, 1> _matB0;
    _matB0.setOnes();
    _matB0 *= -1;
    Eigen::Matrix<double, 3, 1> _matX0;
    _matX0.setZero();

    int laserCloudNonFeatureStackNum = laserCloudNonFeature->points.size();
    for (int i = 0; i < laserCloudNonFeatureStackNum; i++) {
        _pointOri = laserCloudNonFeature->points[i];
        MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
        int id = map_manager->FindUsedNonFeatureMap(
            &_pointSel, laserCenWidth_last, laserCenHeight_last,
            laserCenDepth_last);

        if (id == 5000)
            continue;

        if (std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||
            std::isnan(_pointSel.z))
            continue;

        if (GlobalNonFeatureMap[id].points.size() > 100) {
            NonFeatureKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd,
                                               _pointSearchSqDis);
            if (_pointSearchSqDis[4] < 1 * thres_dist) {
                for (int j = 0; j < 5; j++) {
                    _matA0(j, 0) =
                        GlobalNonFeatureMap[id].points[_pointSearchInd[j]].x;
                    _matA0(j, 1) =
                        GlobalNonFeatureMap[id].points[_pointSearchInd[j]].y;
                    _matA0(j, 2) =
                        GlobalNonFeatureMap[id].points[_pointSearchInd[j]].z;
                }
                _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

                float pa = _matX0(0, 0);
                float pb = _matX0(1, 0);
                float pc = _matX0(2, 0);
                float pd = 1;

                float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (std::fabs(pa * GlobalNonFeatureMap[id]
                                           .points[_pointSearchInd[j]]
                                           .x +
                                  pb * GlobalNonFeatureMap[id]
                                           .points[_pointSearchInd[j]]
                                           .y +
                                  pc * GlobalNonFeatureMap[id]
                                           .points[_pointSearchInd[j]]
                                           .z +
                                  pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    auto* e = Cost_NonFeature_ICP::Create(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd,
                        Eigen::Matrix<double, 1, 1>(1 /
                                                    IMUIntegrator::lidar_m));
                    edges.push_back(e);
                    vNonFeatures.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd);
                    vNonFeatures.back().ComputeError(m4d);

                    continue;
                }
            }
        }

        if (laserCloudNonFeatureLocal->points.size() > 20) {
            kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2,
                                        _pointSearchSqDis2);
            if (_pointSearchSqDis2[4] < 1 * thres_dist) {
                for (int j = 0; j < 5; j++) {
                    _matA0(j, 0) =
                        laserCloudNonFeatureLocal->points[_pointSearchInd2[j]]
                            .x;
                    _matA0(j, 1) =
                        laserCloudNonFeatureLocal->points[_pointSearchInd2[j]]
                            .y;
                    _matA0(j, 2) =
                        laserCloudNonFeatureLocal->points[_pointSearchInd2[j]]
                            .z;
                }
                _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

                float pa = _matX0(0, 0);
                float pb = _matX0(1, 0);
                float pc = _matX0(2, 0);
                float pd = 1;

                float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (std::fabs(pa * laserCloudNonFeatureLocal
                                           ->points[_pointSearchInd2[j]]
                                           .x +
                                  pb * laserCloudNonFeatureLocal
                                           ->points[_pointSearchInd2[j]]
                                           .y +
                                  pc * laserCloudNonFeatureLocal
                                           ->points[_pointSearchInd2[j]]
                                           .z +
                                  pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    auto* e = Cost_NonFeature_ICP::Create(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd,
                        Eigen::Matrix<double, 1, 1>(1 /
                                                    IMUIntegrator::lidar_m));
                    edges.push_back(e);
                    vNonFeatures.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd);
                    vNonFeatures.back().ComputeError(m4d);
                }
            }
        }
    }
}

void LidarModule::MapIncrementLocal(
    const pcl::PointCloud<PointType>::Ptr& laserCloudCornerStack,
    const pcl::PointCloud<PointType>::Ptr& laserCloudSurfStack,
    const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureStack,
    const Eigen::Matrix4d& transformTobeMapped) {
    int laserCloudCornerStackNum = laserCloudCornerStack->points.size();
    int laserCloudSurfStackNum = laserCloudSurfStack->points.size();
    int laserCloudNonFeatureStackNum = laserCloudNonFeatureStack->points.size();

    PointType pointSel;
    PointType pointSel2;

    size_t Id = localMapID % localMapWindowSize;
    localCornerMap[Id]->clear();
    localSurfMap[Id]->clear();
    localNonFeatureMap[Id]->clear();
    for (int i = 0; i < laserCloudCornerStackNum; i++) {
        MAP_MANAGER::pointAssociateToMap(&laserCloudCornerStack->points[i],
                                         &pointSel, transformTobeMapped);
        localCornerMap[Id]->push_back(pointSel);
    }
    for (int i = 0; i < laserCloudSurfStackNum; i++) {
        MAP_MANAGER::pointAssociateToMap(&laserCloudSurfStack->points[i],
                                         &pointSel2, transformTobeMapped);
        localSurfMap[Id]->push_back(pointSel2);
    }
    for (int i = 0; i < laserCloudNonFeatureStackNum; i++) {
        MAP_MANAGER::pointAssociateToMap(&laserCloudNonFeatureStack->points[i],
                                         &pointSel2, transformTobeMapped);
        localNonFeatureMap[Id]->push_back(pointSel2);
    }

    localMapID++;

    laserCloudCornerFromLocal->clear();
    laserCloudSurfFromLocal->clear();
    laserCloudNonFeatureFromLocal->clear();
    for (int i = 0; i < localMapWindowSize; i++) {
        *laserCloudCornerFromLocal += *localCornerMap[i];
        *laserCloudSurfFromLocal += *localSurfMap[i];
        *laserCloudNonFeatureFromLocal += *localNonFeatureMap[i];
    }
    pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>());
    downSizeFilterCorner.setInputCloud(laserCloudCornerFromLocal);
    downSizeFilterCorner.filter(*temp);
    laserCloudCornerFromLocal = temp;
    pcl::PointCloud<PointType>::Ptr temp2(new pcl::PointCloud<PointType>());
    downSizeFilterSurf.setInputCloud(laserCloudSurfFromLocal);
    downSizeFilterSurf.filter(*temp2);
    laserCloudSurfFromLocal = temp2;
    pcl::PointCloud<PointType>::Ptr temp3(new pcl::PointCloud<PointType>());
    downSizeFilterNonFeature.setInputCloud(laserCloudNonFeatureFromLocal);
    downSizeFilterNonFeature.filter(*temp3);
    laserCloudNonFeatureFromLocal = temp3;
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

    double para_v[v_size][3];
    double para_r[3];
    double para_ba[3];
    double para_bg[3];
    double para_scale[1];

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
    for (int i = 0; i < v_size; i++) {
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
        auto iter = frames.begin();
        auto iter_next = frames.begin();
        std::advance(iter, i - 1);
        std::advance(iter_next, i);

        Eigen::Matrix3d exRlbj =
            (*iter_next)->ExT_.block<3, 3>(0, 0).transpose();
        Eigen::Vector3d exPlbj =
            (-exRlbj * (*iter_next)->ExT_.block<3, 1>(0, 3));

        Eigen::Matrix3d exRlbi = (*iter)->ExT_.block<3, 3>(0, 0).transpose();
        Eigen::Vector3d exPlbi = (-exRlbi * (*iter)->ExT_.block<3, 1>(0, 3));

        Eigen::Vector3d pi = (*iter)->P_ + (*iter)->Q_ * exPlbi;
        Sophus::SO3d SO3_Ri(Eigen::Quaterniond((*iter)->Q_ * exRlbi)
                                .normalized()
                                .toRotationMatrix());
        Eigen::Vector3d ri = SO3_Ri.log();
        Eigen::Vector3d pj = (*iter_next)->P_ + (*iter_next)->Q_ * exPlbj;
        Sophus::SO3d SO3_Rj(Eigen::Quaterniond((*iter_next)->Q_ * exRlbj)
                                .normalized()
                                .toRotationMatrix());
        Eigen::Vector3d rj = SO3_Rj.log();

        problem.AddResidualBlock(
            Cost_Initialization_IMU::Create(
                (*iter_next)->imuIntegrator, ri, rj, pj - pi,
                Eigen::LLT<Eigen::Matrix<double, 9, 9>>(
                    (*iter_next)
                        ->imuIntegrator.GetCovariance()
                        .block<9, 9>(0, 0)
                        .inverse())
                    .matrixL()
                    .transpose()),
            nullptr, para_r, para_v[i - 1], para_v[i], para_ba, para_bg,
            para_scale);
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
