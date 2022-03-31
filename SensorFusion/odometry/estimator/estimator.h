#pragma once

#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <vector>

#include "../imu_integrator.h"
#include "frame.h"
#include "module.h"

namespace SensorFusion {

class Estimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Estimator>;
    using IntFramePtr = std::pair<std::pair<int, int>, Frame::Ptr>;
    Estimator(const int Ex_Mode);
    ~Estimator() {}

    void Estimate();

    std::vector<Module::Ptr> modules;

    std::string logger_flag = "!estimator! : ";

private:
    bool imuInitialize();
    void initializeOtherModule(const Module::Ptr& init_module);
    bool dataSynchronize();
    void preProcess();
    void postProcess();
    void optimization();
    void mergeWindow();
    void addImuResidualBlock();
    void addMargResidualBlock();
    void marginalization();
    void slideWindow();

private:
    ProblemPtr problem;
    int max_iters = 5;
    int max_num_iterations = 10;
    int max_num_threads = 6;

    MarginalizationInfo* last_marginalization_info = nullptr;
    std::vector<double*> last_marginalization_parameter_blocks;

    std::vector<IntFramePtr> all_frames;
    std::vector<IMUIntegrator> all_imu_preintegrators;
    Eigen::Vector3d gravity_vector;

    ceres::Problem::EvaluateOptions evaluate_options;
    std::vector<ceres::ResidualBlockId> imu_residual_block_ids;

    int imu_initialized = 0;
};

}  // namespace SensorFusion
