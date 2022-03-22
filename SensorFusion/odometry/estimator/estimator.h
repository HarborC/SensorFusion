#pragma once

#include <unistd.h>
#include <algorithm>

#include "../feature_manager.h"
#include "../global.h"
#include "../image_tracker.h"
#include "../imu_integrator.h"
#include "../map_manager.h"
#include "../type.h"
#include "../utils/lidar_utils.h"
#include "ceresfunc.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "../nano_gicp/lsq_registration.hpp"
#include "../nano_gicp/nano_gicp.hpp"
#include "../nano_gicp/nanoflann.hpp"

namespace SensorFusion {

using ProblemPtr = std::shared_ptr<ceres::Problem>;
/** \brief slide window size */
static const int MAX_SLIDE_WINDOW_SIZE = 20;
static const int SLIDE_WINDOW_SIZE = 3;
static const int SIZE_POSE = 6;
static const int SIZE_SPEEDBIAS = 9;

enum SensorFlag { UNKNOWN = 0, CAMERA = 1, LIDAR = 2 };

/** \brief frame structure */
struct Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Frame>;
    Frame() {
        sensorType = SensorFlag::UNKNOWN;
        pre_imu_enabled = false;
        P = Eigen::Vector3d::Zero();
        Q = Eigen::Quaterniond::Identity();
        V = Eigen::Vector3d::Zero();
        bg = Eigen::Vector3d::Zero();
        ba = Eigen::Vector3d::Zero();
    }
    Frame(const Frame::Ptr& frame) {
        timeStamp = frame->timeStamp;
        sensorType = frame->sensorType;
        imuIntegrator = frame->imuIntegrator;
        pre_imu_enabled = frame->pre_imu_enabled;
        sensor_id = frame->sensor_id;
        P = frame->P;
        Q = frame->Q;
        V = frame->V;
        bg = frame->bg;
        ba = frame->ba;
        ground_plane_coeff = frame->ground_plane_coeff;
        ExT_ = frame->ExT_;
        P_ = frame->P_;
        Q_ = frame->Q_;
    }
    virtual ~Frame() {}
    double timeStamp;
    int sensorType;

    IMUIntegrator imuIntegrator;
    bool pre_imu_enabled;
    std::string sensor_id;
    Eigen::Vector3d P;
    Eigen::Quaterniond Q;
    Eigen::Vector3d V;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    Eigen::VectorXf ground_plane_coeff;
    Eigen::Matrix4d ExT_;  // Transformation from Sensor to Body
    Eigen::Vector3d P_;
    Eigen::Quaterniond Q_;
};
using IntFramePtr = std::pair<std::pair<int, int>, Frame::Ptr>;

/** \brief camera frame structure */
struct CameraFrame : public Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<CameraFrame>;
    CameraFrame() : Frame() { sensorType = SensorFlag::CAMERA; }
    ~CameraFrame() override = default;

    ImageTrackerResult::Ptr trackResult;
};

/** \brief lidar frame structure */
struct LidarFrame : public Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<LidarFrame>;
    LidarFrame() : Frame() { sensorType = SensorFlag::LIDAR; }
    ~LidarFrame() override = default;

    pcl::PointCloud<PointType>::Ptr laserCloud;
};

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

/** \brief lidar module structure */
struct LidarModule : public Module {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<LidarModule>;

    LidarModule();
    ~LidarModule() override = default;

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
    bool getFineSolveFlag();
    void slideWindow(const Module::Ptr& prime_module = nullptr);

    void processPointToLine(
        std::vector<ceres::CostFunction*>& edges,
        std::vector<FeatureLine>& vLineFeatures,
        const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
        const pcl::PointCloud<PointType>::Ptr& laserCloudCornerMap,
        const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
        const Eigen::Matrix4d& m4d);

    void processPointToPlan(
        std::vector<ceres::CostFunction*>& edges,
        std::vector<FeaturePlan>& vPlanFeatures,
        const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
        const pcl::PointCloud<PointType>::Ptr& laserCloudSurfMap,
        const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
        const Eigen::Matrix4d& m4d);

    void processPointToPlanVec(
        std::vector<ceres::CostFunction*>& edges,
        std::vector<FeaturePlanVec>& vPlanFeatures,
        const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
        const pcl::PointCloud<PointType>::Ptr& laserCloudSurfMap,
        const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
        const Eigen::Matrix4d& m4d);

    void processNonFeatureICP(
        std::vector<ceres::CostFunction*>& edges,
        std::vector<FeatureNon>& vNonFeatures,
        const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
        const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureLocal,
        const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
        const Eigen::Matrix4d& m4d);

    void MapIncrementLocal(
        const pcl::PointCloud<PointType>::Ptr& laserCloudCornerStack,
        const pcl::PointCloud<PointType>::Ptr& laserCloudSurfStack,
        const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureStack,
        const Eigen::Matrix4d& transformTobeMapped);

    void RemoveLidarDistortion(pcl::PointCloud<PointType>::Ptr& cloud,
                               const Eigen::Matrix3d& dRlc,
                               const Eigen::Vector3d& dtlc);

    [[noreturn]] void threadMapIncrement();

    void getBackLidarPose();
    void mergeSlidePointCloud();

    /** \brief store map points */
    MAP_MANAGER::Ptr map_manager;

    using MeasurementPtr = LidarFeatureResult::Ptr;
    tbb::concurrent_bounded_queue<MeasurementPtr>* data_queue;
    MeasurementPtr curr_data;

    Eigen::Matrix4d transformForMap;

    float filter_surf = 0.4, filter_corner = 0.2;

    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerLast;
    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfLast;
    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudNonFeatureLast;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromLocal;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromLocal;
    pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureFromLocal;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerForMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfForMap;
    pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureForMap;

    int init_ground_count;
    pcl::PointCloud<PointType>::Ptr initGroundCloud;

    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerStack;
    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfStack;
    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudNonFeatureStack;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromLocal;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromLocal;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeNonFeatureFromLocal;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterNonFeature;

    std::mutex mtx_Map;
    std::thread threadMap;

    pcl::KdTreeFLANN<PointType> CornerKdMap[10000];
    pcl::KdTreeFLANN<PointType> SurfKdMap[10000];
    pcl::KdTreeFLANN<PointType> NonFeatureKdMap[10000];

    pcl::PointCloud<PointType> GlobalSurfMap[10000];
    pcl::PointCloud<PointType> GlobalCornerMap[10000];
    pcl::PointCloud<PointType> GlobalNonFeatureMap[10000];

    int laserCenWidth_last = 10;
    int laserCenHeight_last = 5;
    int laserCenDepth_last = 10;

    static const int localMapWindowSize = 50;
    int localMapID = 0;
    pcl::PointCloud<PointType>::Ptr localCornerMap[localMapWindowSize];
    pcl::PointCloud<PointType>::Ptr localSurfMap[localMapWindowSize];
    pcl::PointCloud<PointType>::Ptr localNonFeatureMap[localMapWindowSize];

    std::vector<std::vector<FeatureLine>> vLineFeatures;
    std::vector<std::vector<FeaturePlan>> vPlanFeatures;
    std::vector<std::vector<FeatureNon>> vNonFeatures;

    int map_update_ID = 0;
    int map_skip_frame = 2;  // every map_skip_frame frame update map
    double plan_weight_tan = 0.0;
    double thres_dist = 1.0;
    bool to_be_used = false;
    Eigen::Quaterniond q_before_opti;
    Eigen::Vector3d t_before_opti;

    pcl::PointCloud<PointType>::Ptr get_corner_map() {
        return map_manager->get_corner_map();
    }
    pcl::PointCloud<PointType>::Ptr get_surf_map() {
        return map_manager->get_surf_map();
    }
    pcl::PointCloud<PointType>::Ptr get_nonfeature_map() {
        return map_manager->get_nonfeature_map();
    }
    pcl::PointCloud<PointType>::Ptr get_init_ground_cloud() {
        return initGroundCloud;
    }

    std::string logger_flag = "!estimator_lidar_module! : ";

    // Test
    nano_gicp::NanoGICP<PointType, PointType> gicpScan2Scan;
    nano_gicp::NanoGICP<PointType, PointType> gicpScan2Map;
};

class Estimator {
public:
    enum SolverFlag { INITIAL, NON_LINEAR };

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Estimator>;
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

bool tryImuAlignment(std::list<Frame::Ptr>& frames, double& scale,
                     Eigen::Vector3d& GravityVector);
bool tryImuAlignment(const std::list<ImuData::Ptr>& imu_meas,
                     Eigen::Vector3d& gyroBias, Eigen::Vector3d& GravityVector);
}  // namespace SensorFusion
