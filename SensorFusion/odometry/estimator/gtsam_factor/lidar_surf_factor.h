#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam {

class LidarPose3PlaneNormFactor
    : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
private:
    gtsam::Vector3 curr_point_;
    gtsam::Vector3 plane_unit_norm_;
    double negative_OA_dot_norm_;
    gtsam::Pose3 Tbl_;

public:
    LidarPose3PlaneNormFactor(gtsam::Key pose_key, gtsam::Pose3 Tbl,
                              gtsam::Vector3 curr_point,
                              gtsam::Vector3 plane_unit_norm,
                              double negative_OA_dot_norm,
                              gtsam::SharedNoiseModel noise_model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise_model, pose_key),
          Tbl_(Tbl),
          curr_point_(curr_point),
          plane_unit_norm_(plane_unit_norm),
          negative_OA_dot_norm_(negative_OA_dot_norm) {}

    gtsam::Vector evaluateError(
        const Pose3& pose, boost::optional<Matrix&> H1 = boost::none) const {
        gtsam::Point3 cp(curr_point_(0), curr_point_(1), curr_point_(2));
        gtsam::Point3 norm(plane_unit_norm_(0), plane_unit_norm_(1),
                           plane_unit_norm_(2));

        gtsam::Matrix36 Dpose;
        gtsam::Point3 point_i = Tbl_.transform_from(cp);
        gtsam::Point3 point_w = pose.transform_from(point_i, H1 ? &Dpose : 0);

        gtsam::Vector1 residual;
        gtsam::Matrix13 Dnorm, Dpoint;
        residual(0) = norm.dot(point_w, Dnorm, Dpoint) + negative_OA_dot_norm_;

        // if we need jaccobians
        if (H1) {
            H1->resize(1, 6);

            *H1 << Dpoint * Dpose;
        }

        return residual;
    }
};

class LidarPose3PlaneFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
private:
    gtsam::Vector3 curr_point_;
    gtsam::Vector3 last_point_j_, last_point_l_, last_point_m_;
    gtsam::Vector3 ljm_norm_;
    gtsam::Pose3 Tbl_;

public:
    LidarPose3PlaneFactor(gtsam::Key pose_key, gtsam::Pose3 Tbl,
                          gtsam::Vector3 curr_point,
                          gtsam::Vector3 last_point_j,
                          gtsam::Vector3 last_point_l,
                          gtsam::Vector3 last_point_m, const double& s,
                          gtsam::SharedNoiseModel noise_model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise_model, pose_key),
          Tbl_(Tbl),
          curr_point_(curr_point),
          last_point_j_(last_point_j),
          last_point_l_(last_point_l),
          last_point_m_(last_point_m) {
        ljm_norm_ =
            (last_point_j - last_point_l).cross(last_point_j - last_point_m);
        ljm_norm_.normalize();
    }

    gtsam::Vector evaluateError(
        const Pose3& pose, boost::optional<Matrix&> H1 = boost::none) const {
        gtsam::Point3 cp(curr_point_(0), curr_point_(1), curr_point_(2));
        gtsam::Point3 lpj(last_point_j_(0), last_point_j_(1), last_point_j_(2));
        gtsam::Point3 ljm(ljm_norm_(0), ljm_norm_(1), ljm_norm_(2));

        gtsam::Matrix36 Dpose;
        gtsam::Point3 point_i = Tbl_.transform_from(cp);
        gtsam::Point3 lp = pose.transform_from(point_i, H1 ? &Dpose : 0);

        gtsam::Vector1 residual;
        gtsam::Point3 lpij = lp - lpj;
        gtsam::Matrix13 Dnorm, Dpoint;
        residual(0) = lpij.dot(ljm, Dpoint, Dnorm);

        // if we need jaccobians
        if (H1) {
            H1->resize(1, 6);

            *H1 << Dpoint * Dpose;
        }

        return residual;
    }
};

}  // namespace gtsam