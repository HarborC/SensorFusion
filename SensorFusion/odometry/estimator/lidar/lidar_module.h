#pragma once

#include "../../type.h"
#include "../module.h"
#include "fast_gicp/gicp_utils.hpp"
#include "lidar_feature.h"
#include "lidar_frame.h"
#include "map_manager.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "../../nano_gicp/lsq_registration.hpp"
#include "../../nano_gicp/nano_gicp.hpp"
#include "../../nano_gicp/nanoflann.hpp"

namespace SensorFusion {

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
    void addGtamFactor();

    void processPointToLine(
        std::vector<ceres::CostFunction*>& edges,
        std::vector<FeatureLine>& vLineFeatures,
        const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
        const pcl::PointCloud<PointType>::Ptr& laserCloudCornerMap,
        const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
        const Eigen::Matrix4d& m4d);

    void processPointToPlan(
        std::vector<ceres::CostFunction*>& edges,
        std::vector<FeaturePlan>& vPlanFeature,
        const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
        const pcl::PointCloud<PointType>::Ptr& laserCloudSurfMap,
        const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
        const Eigen::Matrix4d& m4d);

    void processPointToPlanVec(
        std::vector<ceres::CostFunction*>& edges,
        std::vector<FeaturePlanVec>& vPlanFeature,
        const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
        const pcl::PointCloud<PointType>::Ptr& laserCloudSurfMap,
        const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
        const Eigen::Matrix4d& m4d);

    void processNonFeatureICP(
        std::vector<ceres::CostFunction*>& edges,
        std::vector<FeatureNon>& vNonFeature,
        const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
        const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureLocal,
        const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
        const Eigen::Matrix4d& m4d);

    void processPointGICP(
        std::vector<ceres::CostFunction*>& edges,
        std::vector<FeatureGICP>& vGICPFeature,
        const pcl::PointCloud<PointType>::Ptr& laserCloud,
        const std::vector<Eigen::Matrix4d,
                          Eigen::aligned_allocator<Eigen::Matrix4d>>& covs_scan,
        const fast_gicp::GaussianVoxelMap<PointType>::Ptr& voxelmap_local,
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

    std::vector<std::vector<FeatureGICP>> vGICPFeatures;

    fast_gicp::GaussianVoxelMap<PointType>::Ptr voxelmapLocal;

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

}  // namespace SensorFusion