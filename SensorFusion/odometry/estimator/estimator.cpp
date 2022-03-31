#include "estimator.h"
#include "../global.h"
#include "ceres_factor/imu_preintegration_factor.h"
#include "sensor_flag.h"

namespace SensorFusion {

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

        // 添加residuals
        for (size_t i = 0; i < modules.size(); i++) {
            modules[i]->addResidualBlock(iterOpt);
        }
        addImuResidualBlock();
        addMargResidualBlock();

        problem->GetParameterBlocks(&(evaluate_options.parameter_blocks));

        bool show_opt = true;

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

}  // namespace SensorFusion
