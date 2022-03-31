#include "lidar_module.h"
#include "../../global.h"
#include "../ceres_factor/lidar_corner_factor.h"
#include "../ceres_factor/lidar_gicp_factor.h"
#include "../ceres_factor/lidar_icp_factor.h"
#include "../ceres_factor/lidar_surf_factor.h"

#include <unistd.h>
#include <algorithm>

namespace SensorFusion {

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
            new_frame->filterLaserCloud = curr_data->filter_points;
            new_frame->filterCovs = curr_data->filter_points_covariances;
            new_frame->timeStamp = curr_data->timestamp;
            new_frame->sensor_id = curr_data->sensor_id;
            curr_timestamp = curr_data->timestamp;

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
        new_frame->filterLaserCloud = curr_data->filter_points;
        new_frame->filterCovs = curr_data->filter_points_covariances;
        new_frame->timeStamp = curr_data->timestamp;
        new_frame->sensor_id = curr_data->sensor_id;
        curr_timestamp = curr_data->timestamp;

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

                vLineFeatures[0].clear();
                vPlanFeatures[0].clear();
                vNonFeatures[0].clear();

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

                bool show_opt = true;

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

    frame_curr->P_ = frame_curr->Q * exPbl + frame_curr->P;
    frame_curr->Q_ = frame_curr->Q * exRbl;
    frame_curr->ExT_ = ex_pose[lidar_id];

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

        {
            pcl::PointCloud<PointType>::Ptr laserCloudFromLocal(
                new pcl::PointCloud<PointType>);
            *laserCloudFromLocal += *laserCloudCornerFromLocal;
            *laserCloudFromLocal += *laserCloudSurfFromLocal;
            *laserCloudFromLocal += *laserCloudNonFeatureFromLocal;

            nanoflann::KdTreeFLANN<PointType> nanokdtree;
            std::vector<Eigen::Matrix4d,
                        Eigen::aligned_allocator<Eigen::Matrix4d>>
                covariances;

            TicToc t0;
            t0.tic();
            {
                fast_gicp::calculate_covariances(laserCloudFromLocal,
                                                 nanokdtree, covariances);
            }
            logger->recordLogger(logger_flag, t0.toc(), frame_count,
                                 "fast_gicp::calculate_covariances");

            t0.tic();
            {
                voxelmapLocal.reset(
                    new fast_gicp::GaussianVoxelMap<PointType>(1.0));
                voxelmapLocal->create_voxelmap(*laserCloudFromLocal,
                                               covariances);
            }
            logger->recordLogger(logger_flag, t0.toc(), frame_count,
                                 "voxelmapLocal->create_voxelmap");
            t0.tic();
        }

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
        vLineFeatures.clear();
        vLineFeatures.resize(windowSize);
        for (auto& v : vLineFeatures) {
            v.reserve(2000);
        }

        vPlanFeatures.clear();
        vPlanFeatures.resize(windowSize);
        for (auto& v : vPlanFeatures) {
            v.reserve(2000);
        }

        vNonFeatures.clear();
        vNonFeatures.resize(windowSize);
        for (auto& v : vNonFeatures) {
            v.reserve(2000);
        }

        vGICPFeatures.clear();
        vGICPFeatures.resize(windowSize);
        for (auto& v : vGICPFeatures) {
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
    *slide_pointcloud_corner += *laserCloudCornerStack[slide_window_size - 1];
    *slide_pointcloud_surf += *laserCloudSurfStack[slide_window_size - 1];
    *slide_pointcloud_nonfeature +=
        *laserCloudNonFeatureStack[slide_window_size - 1];

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

    {
        map_viewer.addLocalMap(laserCloudCornerFromLocal, "local_corner", 110,
                               0, 0);
        map_viewer.addLocalMap(laserCloudSurfFromLocal, "local_surf", 0, 110,
                               0);
        size_t Id = (localMapID - 1) % localMapWindowSize;
        map_viewer.addCurrPoints(localCornerMap[Id], "curr_corner", 200, 0, 0);
        map_viewer.addCurrPoints(localSurfMap[Id], "curr_surf", 0, 200, 0);
        map_viewer.addPose((*curr_frame)->timeStamp, transformForMap);
    }
    locker.unlock();
}

void LidarModule::postProcess() {
    if (!to_be_used)
        return;
}

void LidarModule::addResidualBlock(int iterOpt) {
    if (!to_be_used)
        return;

    auto curr_frame_li =
        std::dynamic_pointer_cast<LidarFrame>(window_frames.back());
    std::string lidar_id = curr_frame_li->sensor_id;
    q_before_opti = curr_frame_li->Q;
    t_before_opti = curr_frame_li->P;

    Eigen::Matrix<double, 3, 3> exRbl = ex_pose[lidar_id].block<3, 3>(0, 0);
    Eigen::Matrix<double, 3, 1> exPbl = ex_pose[lidar_id].block<3, 1>(0, 3);

    int windowSize = window_frames.size();

    if (curr_frame_li->filterLaserCloud) {
        std::vector<std::vector<ceres::CostFunction*>> edgesGICP(windowSize);
        for (int f = 0; f < windowSize; ++f) {
            auto frame_curr = window_frames.begin();
            std::advance(frame_curr, f);

            auto frame_curr_li =
                std::dynamic_pointer_cast<LidarFrame>(*frame_curr);
            Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
            transformTobeMapped.topLeftCorner(3, 3) = frame_curr_li->Q * exRbl;
            transformTobeMapped.topRightCorner(3, 1) =
                frame_curr_li->Q * exPbl + frame_curr_li->P;

            vGICPFeatures[f].clear();
            processPointGICP(
                edgesGICP[f], vGICPFeatures[f], frame_curr_li->filterLaserCloud,
                frame_curr_li->filterCovs, voxelmapLocal, transformTobeMapped);
        }

        // create huber loss function
        ceres::LossFunction* loss_function = NULL;
        loss_function = NULL;

        for (int f = 0; f < windowSize; ++f) {
            for (auto e : edgesGICP[f]) {
                auto re_id = problem->AddResidualBlock(
                    e, loss_function, para_Pose[f], para_Ex_Pose[lidar_id]);
            }
        }
    } else {
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

            vLineFeatures[f].clear();
            vPlanFeatures[f].clear();
            vNonFeatures[f].clear();

            std::thread threads[3];
            threads[0] = std::thread(
                &LidarModule::processPointToLine, this, std::ref(edgesLine[f]),
                std::ref(vLineFeatures[f]), std::ref(laserCloudCornerStack[f]),
                std::ref(laserCloudCornerFromLocal),
                std::ref(kdtreeCornerFromLocal), std::ref(transformTobeMapped));

            threads[1] = std::thread(
                &LidarModule::processPointToPlan, this, std::ref(edgesPlan[f]),
                std::ref(vPlanFeatures[f]), std::ref(laserCloudSurfStack[f]),
                std::ref(laserCloudSurfFromLocal),
                std::ref(kdtreeSurfFromLocal), std::ref(transformTobeMapped));

            threads[2] =
                std::thread(&LidarModule::processNonFeatureICP, this,
                            std::ref(edgesNon[f]), std::ref(vNonFeatures[f]),
                            std::ref(laserCloudNonFeatureStack[f]),
                            std::ref(laserCloudNonFeatureFromLocal),
                            std::ref(kdtreeNonFeatureFromLocal),
                            std::ref(transformTobeMapped));

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
                            e, loss_function, para_Pose[f],
                            para_Ex_Pose[lidar_id]);
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
                            e, loss_function, para_Pose[f],
                            para_Ex_Pose[lidar_id]);
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
                            e, loss_function, para_Pose[f],
                            para_Ex_Pose[lidar_id]);
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
                            e, loss_function, para_Pose[f],
                            para_Ex_Pose[lidar_id]);
                        residual_block_ids["lidar_corner"].push_back(re_id);
                    }
                    cntFtu++;
                    cntCorner++;
                }

                cntFtu = 0;
                for (auto& e : edgesPlan[f]) {
                    if (vPlanFeatures[f][cntFtu].valid) {
                        auto re_id = problem->AddResidualBlock(
                            e, loss_function, para_Pose[f],
                            para_Ex_Pose[lidar_id]);
                        residual_block_ids["lidar_surf"].push_back(re_id);
                    }
                    cntFtu++;
                    cntSurf++;
                }

                cntFtu = 0;
                for (auto& e : edgesNon[f]) {
                    if (vNonFeatures[f][cntFtu].valid) {
                        auto re_id = problem->AddResidualBlock(
                            e, loss_function, para_Pose[f],
                            para_Ex_Pose[lidar_id]);
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

    if (std::dynamic_pointer_cast<LidarFrame>(window_frames.front())
            ->filterLaserCloud) {
        std::vector<std::vector<ceres::CostFunction*>> edgesGICP(
            slide_win_size);

        for (int f = 0; f < slide_win_size; ++f) {
            auto frame_curr = window_frames.begin();
            std::advance(frame_curr, f);

            auto frame_curr_li =
                std::dynamic_pointer_cast<LidarFrame>(*frame_curr);
            std::string lidar_id = frame_curr_li->sensor_id;
            Eigen::Matrix<double, 3, 3> exRbl =
                ex_pose[lidar_id].block<3, 3>(0, 0);
            Eigen::Matrix<double, 3, 1> exPbl =
                ex_pose[lidar_id].block<3, 1>(0, 3);
            Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
            transformTobeMapped.topLeftCorner(3, 3) = frame_curr_li->Q * exRbl;
            transformTobeMapped.topRightCorner(3, 1) =
                frame_curr_li->Q * exPbl + frame_curr_li->P;

            vGICPFeatures[f].clear();
            processPointGICP(
                edgesGICP[f], vGICPFeatures[f], frame_curr_li->filterLaserCloud,
                frame_curr_li->filterCovs, voxelmapLocal, transformTobeMapped);

            for (auto e : edgesGICP[f]) {
                auto* residual_block_info = new ResidualBlockInfo(
                    e, nullptr,
                    std::vector<double*>{para_Pose[f], para_Ex_Pose[lidar_id]},
                    std::vector<int>{0});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }
    } else {
        std::vector<std::vector<ceres::CostFunction*>> edgesLine(
            slide_win_size);
        std::vector<std::vector<ceres::CostFunction*>> edgesPlan(
            slide_win_size);
        std::vector<std::vector<ceres::CostFunction*>> edgesNon(slide_win_size);

        for (size_t f = 0; f < slide_win_size; f++) {
            auto curr_frame = window_frames.begin();
            std::advance(curr_frame, f);

            std::string lidar_id = (*curr_frame)->sensor_id;
            Eigen::Matrix<double, 3, 3> exRbl =
                ex_pose[lidar_id].block<3, 3>(0, 0);
            Eigen::Matrix<double, 3, 1> exPbl =
                ex_pose[lidar_id].block<3, 1>(0, 3);
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
                std::ref(laserCloudSurfFromLocal),
                std::ref(kdtreeSurfFromLocal), std::ref(transformTobeMapped));

            threads[2] =
                std::thread(&LidarModule::processNonFeatureICP, this,
                            std::ref(edgesNon[f]), std::ref(vNonFeatures[f]),
                            std::ref(laserCloudNonFeatureStack[f]),
                            std::ref(laserCloudNonFeatureFromLocal),
                            std::ref(kdtreeNonFeatureFromLocal),
                            std::ref(transformTobeMapped));

            threads[0].join();
            threads[1].join();
            threads[2].join();
            int cntFtu = 0;
            for (auto& e : edgesLine[f]) {
                if (vLineFeatures[f][cntFtu].valid) {
                    auto* residual_block_info = new ResidualBlockInfo(
                        e, nullptr,
                        std::vector<double*>{para_Pose[f],
                                             para_Ex_Pose[lidar_id]},
                        std::vector<int>{0});
                    marginalization_info->addResidualBlockInfo(
                        residual_block_info);
                }
                cntFtu++;
            }

            cntFtu = 0;
            for (auto& e : edgesPlan[f]) {
                if (vPlanFeatures[f][cntFtu].valid) {
                    auto* residual_block_info = new ResidualBlockInfo(
                        e, nullptr,
                        std::vector<double*>{para_Pose[f],
                                             para_Ex_Pose[lidar_id]},
                        std::vector<int>{0});
                    marginalization_info->addResidualBlockInfo(
                        residual_block_info);
                }
                cntFtu++;
            }

            cntFtu = 0;
            for (auto& e : edgesNon[f]) {
                if (vNonFeatures[f][cntFtu].valid) {
                    auto* residual_block_info = new ResidualBlockInfo(
                        e, nullptr,
                        std::vector<double*>{para_Pose[f],
                                             para_Ex_Pose[lidar_id]},
                        std::vector<int>{0});
                    marginalization_info->addResidualBlockInfo(
                        residual_block_info);
                }
                cntFtu++;
            }
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
    std::vector<FeatureLine>& vLineFeature,
    const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
    const pcl::PointCloud<PointType>::Ptr& laserCloudCornerLocal,
    const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
    const Eigen::Matrix4d& m4d) {
    if (!vLineFeature.empty()) {
        for (const auto& l : vLineFeature) {
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
                    vLineFeature.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        tripod1, tripod2);
                    vLineFeature.back().ComputeError(m4d);

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
                    vLineFeature.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        tripod1, tripod2);
                    vLineFeature.back().ComputeError(m4d);
                }
            }
        }
    }
}

void LidarModule::processPointToPlan(
    std::vector<ceres::CostFunction*>& edges,
    std::vector<FeaturePlan>& vPlanFeature,
    const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
    const pcl::PointCloud<PointType>::Ptr& laserCloudSurfLocal,
    const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
    const Eigen::Matrix4d& m4d) {
    if (!vPlanFeature.empty()) {
        for (const auto& p : vPlanFeature) {
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
                    vPlanFeature.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd);
                    vPlanFeature.back().ComputeError(m4d);

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
                    vPlanFeature.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd);
                    vPlanFeature.back().ComputeError(m4d);
                }
            }
        }
    }
}

void LidarModule::processPointToPlanVec(
    std::vector<ceres::CostFunction*>& edges,
    std::vector<FeaturePlanVec>& vPlanFeature,
    const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
    const pcl::PointCloud<PointType>::Ptr& laserCloudSurfLocal,
    const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
    const Eigen::Matrix4d& m4d) {
    if (!vPlanFeature.empty()) {
        for (const auto& p : vPlanFeature) {
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
                    vPlanFeature.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        point_proj, sqrt_info);
                    vPlanFeature.back().ComputeError(m4d);

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
                    vPlanFeature.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        point_proj, sqrt_info);
                    vPlanFeature.back().ComputeError(m4d);
                }
            }
        }
    }
}

void LidarModule::processNonFeatureICP(
    std::vector<ceres::CostFunction*>& edges,
    std::vector<FeatureNon>& vNonFeature,
    const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
    const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureLocal,
    const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
    const Eigen::Matrix4d& m4d) {
    if (!vNonFeature.empty()) {
        for (const auto& p : vNonFeature) {
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
                    vNonFeature.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd);
                    vNonFeature.back().ComputeError(m4d);

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
                    vNonFeature.emplace_back(
                        Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                        pa, pb, pc, pd);
                    vNonFeature.back().ComputeError(m4d);
                }
            }
        }
    }
}

void LidarModule::processPointGICP(
    std::vector<ceres::CostFunction*>& edges,
    std::vector<FeatureGICP>& vGICPFeature,
    const pcl::PointCloud<PointType>::Ptr& laserCloud,
    const std::vector<Eigen::Matrix4d,
                      Eigen::aligned_allocator<Eigen::Matrix4d>>& covs_scan,
    const fast_gicp::GaussianVoxelMap<PointType>::Ptr& voxelmap_local,
    const Eigen::Matrix4d& m4d) {
    fast_gicp::update_correspondences<PointType>(m4d, laserCloud,
                                                 voxelmap_local, vGICPFeature);

    std::cout << "vGICPFeature.size() : " << vGICPFeature.size() << std::endl;

    edges.clear();
    for (auto feature : vGICPFeature) {
        auto idx0 = feature.first;
        const auto& pt = laserCloud->points[idx0];

        auto* e = Cost_IMU_VGICP::Create(
            feature.second->num_points, feature.second->mean,
            Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0), feature.second->cov,
            covs_scan[idx0], Eigen::Matrix<double, 1, 1>(1));
        edges.push_back(e);
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

void LidarModule::addGtamFactor() {}

}  // namespace SensorFusion