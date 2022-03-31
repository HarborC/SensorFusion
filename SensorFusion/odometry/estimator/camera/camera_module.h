#pragma once

#include "../module.h"
#include "camera_frame.h"
#include "feature_manager.h"

namespace SensorFusion {

/** \brief camera module structure */
struct CameraModule : public Module {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<CameraModule>;
    static const int MAX_CAM_NUM = 10;
    static const int MAX_FEAT_NUM = 200;

    CameraModule() : Module() {
        sensorType = SensorFlag::CAMERA;
        prime_flag = false;
        scale = 1.0;
    }
    ~CameraModule() override = default;

    int initialize();
    double dataSynchronize(const double& timestamp = 0);
    void preProcess();
    void postProcess();
    void vector2double();
    void double2vector();
    void addParameter();
    void addResidualBlock(int iterOpt);
    void marginalization1(
        MarginalizationInfo* last_marginalization_info,
        std::vector<double*>& last_marginalization_parameter_blocks,
        MarginalizationInfo* marginalization_info, int slide_win_size);
    void marginalization2(std::unordered_map<long, double*>& addr_shift,
                          int slide_win_size);
    bool getFineSolveFlag() { return false; };
    void slideWindow(const Module::Ptr& prime_module = nullptr);

    void addGtamFactor(){};

    void triangulate();
    bool relativePose(const std::string& cam_id, Eigen::Matrix3d& relative_R,
                      Eigen::Vector3d& relative_T, int& l);

    using MeasurementPtr = ImageTrackerResult::Ptr;
    tbb::concurrent_bounded_queue<MeasurementPtr>* data_queue;
    MeasurementPtr curr_data;

    double para_Feature[MAX_FEAT_NUM][1];
    FeatureManager f_manager;

    std::string logger_flag = "!estimator_camera_module! : ";
};

}  // namespace SensorFusion