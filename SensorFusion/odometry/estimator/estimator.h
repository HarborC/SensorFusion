#pragma once

#include <unistd.h>
#include <algorithm>

#include "../feature_manager.h"
#include "../image_tracker.h"
#include "../imu_integrator.h"
#include "../map_manager.h"
#include "../type.h"
#include "../utils/lidar_utils.h"
#include "ceresfunc.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

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
    using Ptr = std::shared_ptr<Frame>;
    Frame() { sensorType = SensorFlag::UNKNOWN; }
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
    using Ptr = std::shared_ptr<CameraFrame>;
    CameraFrame() { sensorType = SensorFlag::CAMERA; }
    ~CameraFrame() override = default;

    ImageTrackerResult::Ptr trackResult;
};

/** \brief lidar frame structure */
struct LidarFrame : public Frame {
    using Ptr = std::shared_ptr<LidarFrame>;
    LidarFrame() { sensorType = SensorFlag::LIDAR; }
    ~LidarFrame() override = default;

    pcl::PointCloud<PointType>::Ptr laserCloud;
};

/** \brief module structure */
struct Module {
    using Ptr = std::shared_ptr<Module>;
    virtual ~Module() {}

    virtual bool initialize() = 0;
    virtual double dataSynchronize(const double& timestamp = 0) = 0;
    virtual void preProcess() = 0;
    virtual void vector2double() = 0;
    virtual void double2vector() = 0;
    virtual void addResidualBlock(int iterOpt) = 0;
    virtual void marginalization1(
        MarginalizationInfo* last_marginalization_info,
        std::vector<double*>& last_marginalization_parameter_blocks,
        MarginalizationInfo* marginalization_info, int slide_win_size) = 0;
    virtual void marginalization2(MarginalizationInfo* marginalization_info,
                                  std::vector<double*>& parameter_blocks,
                                  int slide_win_size) = 0;
    virtual bool getFineSolveFlag() = 0;

    void getPreIntegratedImuData(std::vector<ImuData::Ptr>& meas);

    int getMargWindowSize(const double& min_time_stamp) {
        int sum_size = 0;
        auto f = window_frames.begin();
        for (size_t i = 0; i < window_frames.size(); i++) {
            std::advance(f, i);
            if ((*f)->timeStamp < min_time_stamp) {
                sum_size++;
            }
        }
        return sum_size;
    }

    int sensorType;
    bool prime_flag;

    tbb::concurrent_bounded_queue<ImuData::Ptr>* imu_data_queue;
    ImuData::Ptr curr_imu;
    double lastTimestamp = 0;
    double currTimestamp;
    int frame_count = 0;

    ProblemPtr problem;
    std::list<Frame::Ptr> window_frames;
    /** \brief ex_pose means the translation from sensor to body */
    std::unordered_map<std::string, Eigen::Matrix<double, 4, 4>> ex_pose;

    double para_Pose[MAX_SLIDE_WINDOW_SIZE][6];
    double para_SpeedBias[MAX_SLIDE_WINDOW_SIZE][9];
    std::unordered_map<std::string, double*> para_Ex_Pose;
    Eigen::Vector3d curr_ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d curr_bg = Eigen::Vector3d::Zero();
};

/** \brief camera module structure */
struct CameraModule : public Module {
    using Ptr = std::shared_ptr<CameraModule>;
    static const int MAX_CAM_NUM = 10;
    static const int MAX_FEAT_NUM = 200;

    CameraModule() {
        sensorType = SensorFlag::CAMERA;
        prime_flag = false;
    }
    ~CameraModule() override = default;

    bool initialize();
    double dataSynchronize(const double& timestamp = 0);
    void preProcess();
    void vector2double();
    void double2vector();
    void addResidualBlock(int iterOpt);
    void marginalization1(
        MarginalizationInfo* last_marginalization_info,
        std::vector<double*>& last_marginalization_parameter_blocks,
        MarginalizationInfo* marginalization_info, int slide_win_size);
    void marginalization2(MarginalizationInfo* marginalization_info,
                          std::vector<double*>& parameter_blocks,
                          int slide_win_size);
    bool getFineSolveFlag() { return false; };

    void triangulate();
    bool relativePose(const std::string& cam_id, Eigen::Matrix3d& relative_R,
                      Eigen::Vector3d& relative_T, int& l);

    using MeasurementPtr = ImageTrackerResult::Ptr;
    tbb::concurrent_bounded_queue<MeasurementPtr>* data_queue;
    MeasurementPtr curr_data;

    double para_Feature[MAX_FEAT_NUM][1];
    FeatureManager f_manager;
};

/** \brief lidar module structure */
struct LidarModule : public Module {
    using Ptr = std::shared_ptr<LidarModule>;

    LidarModule();
    ~LidarModule() override = default;

    bool initialize();
    double dataSynchronize(const double& timestamp = 0);
    void preProcess();
    void vector2double();
    void double2vector();
    void addResidualBlock(int iterOpt);
    void marginalization1(
        MarginalizationInfo* last_marginalization_info,
        std::vector<double*>& last_marginalization_parameter_blocks,
        MarginalizationInfo* marginalization_info, int slide_win_size);
    void marginalization2(MarginalizationInfo* marginalization_info,
                          std::vector<double*>& parameter_blocks,
                          int slide_win_size);
    bool getFineSolveFlag();

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

    /** \brief store map points */
    MAP_MANAGER::Ptr map_manager;

    using MeasurementPtr = LidarFeatureResult::Ptr;
    tbb::concurrent_bounded_queue<MeasurementPtr>* data_queue;
    MeasurementPtr curr_data;

    Eigen::Matrix4d transformForMap;

    float filter_surf, filter_corner;

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
    std::vector<std::vector<FeaturePlanVec>> vPlanFeatures;
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
};

class Estimator {
public:
    enum SolverFlag { INITIAL, NON_LINEAR };

public:
    using Ptr = std::shared_ptr<Estimator>;
    Estimator(const int Ex_Mode);
    ~Estimator() {}

    void Estimate();

    std::vector<Module::Ptr> modules;

private:
    bool imuInitialize();
    bool dataSynchronize();
    void preProcess();
    void optimization();
    void addImuResidualBlock();
    void addMargResidualBlock();
    void marginalization();

private:
    bool imu_initialized;
    ProblemPtr problem;
    int max_iters = 3;
    int max_num_iterations = 6;
    int max_num_threads = 6;

    MarginalizationInfo* last_marginalization_info = nullptr;
    std::vector<double*> last_marginalization_parameter_blocks;

    Eigen::Vector3d gravity;
    std::vector<IntFramePtr> all_frames;
    std::vector<IMUIntegrator> all_imu_preintegrators;
};

bool tryImuAlignment(std::list<Frame::Ptr>& frames, double& scale,
                     Eigen::Vector3d& GravityVector);
bool tryImuAlignment(std::vector<ImuData::Ptr>& imu_meas,
                     Eigen::Vector3f& gyroBias, Eigen::Matrix3f& Rwg);
}  // namespace SensorFusion
