#pragma once

#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ceres/ceres.h>
#include <tbb/concurrent_queue.h>
#include <Eigen/Core>

#include "../imu_data.h"
#include "ceres_factor/marginalization_factor.h"
#include "frame.h"

namespace SensorFusion {

/** \brief slide window size */
static const int MAX_SLIDE_WINDOW_SIZE = 20;
static const int SLIDE_WINDOW_SIZE = 3;
static const int SIZE_POSE = 6;
static const int SIZE_SPEEDBIAS = 9;

using ProblemPtr = std::shared_ptr<ceres::Problem>;

/** \brief module structure */
struct Module {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Module>;
    enum InitializtionType { NONE = 0, STATIC = 1, DYNAMIC = 2 };

    Module() { prev_frame.reset(new Frame()); }
    virtual ~Module() {}

    virtual int initialize() = 0;
    virtual double dataSynchronize(const double& timestamp = 0) = 0;
    virtual void preProcess() = 0;
    virtual void postProcess() = 0;
    virtual void vector2double() = 0;
    virtual void double2vector() = 0;
    virtual void addParameter() = 0;
    virtual void addResidualBlock(int iterOpt) = 0;
    virtual void marginalization1(
        MarginalizationInfo* last_marginalization_info,
        std::vector<double*>& last_marginalization_parameter_blocks,
        MarginalizationInfo* marginalization_info, int slide_win_size) = 0;
    virtual void marginalization2(std::unordered_map<long, double*>& addr_shift,
                                  int slide_win_size) = 0;
    virtual bool getFineSolveFlag() = 0;
    virtual void slideWindow(const Module::Ptr& prime_module = nullptr) = 0;
    virtual void addGtamFactor() = 0;

    void getPreIntegratedImuData(std::vector<ImuData::Ptr>& meas);

    bool staticInitialize();
    bool dynamicInitialize();

    int getMargWindowSize(const double& min_time_stamp) {
        int sum_size = 0;
        for (size_t i = 0; i < window_frames.size(); i++) {
            auto f = window_frames.begin();
            std::advance(f, i);
            if ((*f)->timeStamp < min_time_stamp) {
                sum_size++;
            } else {
                break;
            }
        }
        return sum_size;
    }

    int sensorType;
    bool prime_flag;

    tbb::concurrent_bounded_queue<ImuData::Ptr>* imu_data_queue;
    ImuData::Ptr curr_imu;
    double last_timestamp = 0;
    double curr_timestamp;
    int frame_count = 0;
    Frame::Ptr prev_frame;
    int slide_window_size;

    ProblemPtr problem;
    std::list<Frame::Ptr> window_frames;
    std::vector<Frame::Ptr> slide_frames;
    /** \brief ex_pose means the translation from sensor to body */
    std::unordered_map<std::string, Eigen::Matrix<double, 4, 4>> ex_pose;

    double para_Pose[MAX_SLIDE_WINDOW_SIZE][6];
    double para_SpeedBias[MAX_SLIDE_WINDOW_SIZE][9];
    std::unordered_map<std::string, double*> para_Ex_Pose;

    Eigen::Matrix<double, 6, 1> velocity = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 3, 1> gyro_bias = Eigen::Matrix<double, 3, 1>::Zero();
    double scale;
    Eigen::Vector3d gravity_vector;
    int imu_initialized = 0;
    std::list<ImuData::Ptr> accumulate_imu_meas;

    // for test
    std::unordered_map<std::string, std::vector<ceres::ResidualBlockId>>
        residual_block_ids;
};

bool tryImuAlignment(std::list<Frame::Ptr>& frames, double& scale,
                     Eigen::Vector3d& GravityVector);
bool tryImuAlignment(const std::list<ImuData::Ptr>& imu_meas,
                     Eigen::Vector3d& gyroBias, Eigen::Vector3d& GravityVector);

}  // namespace SensorFusion