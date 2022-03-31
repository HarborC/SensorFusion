#pragma once

#include <Eigen/Core>
#include "fast_gicp/gicp_voxel.hpp"

namespace SensorFusion {

/** \brief point to line feature */
struct FeatureLine {
    Eigen::Vector3d pointOri;
    Eigen::Vector3d lineP1;
    Eigen::Vector3d lineP2;
    double error;
    bool valid;
    FeatureLine(Eigen::Vector3d po, Eigen::Vector3d p1, Eigen::Vector3d p2);
    double ComputeError(const Eigen::Matrix4d& pose);
};

/** \brief point to plan feature */
struct FeaturePlan {
    Eigen::Vector3d pointOri;
    double pa;
    double pb;
    double pc;
    double pd;
    double error;
    bool valid;
    FeaturePlan(const Eigen::Vector3d& po, const double& pa_, const double& pb_,
                const double& pc_, const double& pd_);
    double ComputeError(const Eigen::Matrix4d& pose);
};

/** \brief point to plan feature */
struct FeaturePlanVec {
    Eigen::Vector3d pointOri;
    Eigen::Vector3d pointProj;
    Eigen::Matrix3d sqrt_info;
    double error;
    bool valid;
    FeaturePlanVec(const Eigen::Vector3d& po, const Eigen::Vector3d& p_proj,
                   Eigen::Matrix3d sqrt_info_);
    double ComputeError(const Eigen::Matrix4d& pose);
};

/** \brief non feature */
struct FeatureNon {
    Eigen::Vector3d pointOri;
    double pa;
    double pb;
    double pc;
    double pd;
    double error;
    bool valid;
    FeatureNon(const Eigen::Vector3d& po, const double& pa_, const double& pb_,
               const double& pc_, const double& pd_);
    double ComputeError(const Eigen::Matrix4d& pose);
};

using FeatureGICP = std::pair<int, fast_gicp::GaussianVoxel::Ptr>;

}  // namespace SensorFusion