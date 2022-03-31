#include "lidar_feature.h"

namespace SensorFusion {

FeatureLine::FeatureLine(Eigen::Vector3d po, Eigen::Vector3d p1,
                         Eigen::Vector3d p2)
    : pointOri(std::move(po)), lineP1(std::move(p1)), lineP2(std::move(p2)) {
    valid = false;
    error = 0;
}

double FeatureLine::ComputeError(const Eigen::Matrix4d& pose) {
    Eigen::Vector3d P_to_Map =
        pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
    double l12 = std::sqrt((lineP1(0) - lineP2(0)) * (lineP1(0) - lineP2(0)) +
                           (lineP1(1) - lineP2(1)) * (lineP1(1) - lineP2(1)) +
                           (lineP1(2) - lineP2(2)) * (lineP1(2) - lineP2(2)));
    double a012 =
        std::sqrt(((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) -
                   (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1))) *
                      ((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) -
                       (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1))) +
                  ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) -
                   (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2))) *
                      ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) -
                       (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2))) +
                  ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) -
                   (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2))) *
                      ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) -
                       (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2))));
    error = a012 / l12;
}

FeaturePlan::FeaturePlan(const Eigen::Vector3d& po, const double& pa_,
                         const double& pb_, const double& pc_,
                         const double& pd_)
    : pointOri(po), pa(pa_), pb(pb_), pc(pc_), pd(pd_) {
    valid = false;
    error = 0;
}

double FeaturePlan::ComputeError(const Eigen::Matrix4d& pose) {
    Eigen::Vector3d P_to_Map =
        pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
    error = pa * P_to_Map(0) + pb * P_to_Map(1) + pc * P_to_Map(2) + pd;
}

FeaturePlanVec::FeaturePlanVec(const Eigen::Vector3d& po,
                               const Eigen::Vector3d& p_proj,
                               Eigen::Matrix3d sqrt_info_)
    : pointOri(po), pointProj(p_proj), sqrt_info(sqrt_info_) {
    valid = false;
    error = 0;
}

double FeaturePlanVec::ComputeError(const Eigen::Matrix4d& pose) {
    Eigen::Vector3d P_to_Map =
        pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
    error = (P_to_Map - pointProj).norm();
}

FeatureNon::FeatureNon(const Eigen::Vector3d& po, const double& pa_,
                       const double& pb_, const double& pc_, const double& pd_)
    : pointOri(po), pa(pa_), pb(pb_), pc(pc_), pd(pd_) {
    valid = false;
    error = 0;
}

double FeatureNon::ComputeError(const Eigen::Matrix4d& pose) {
    Eigen::Vector3d P_to_Map =
        pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
    error = pa * P_to_Map(0) + pb * P_to_Map(1) + pc * P_to_Map(2) + pd;
}

}  // namespace SensorFusion