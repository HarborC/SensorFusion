#pragma once
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam {

class LidarPose3ICPFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
private:
    gtsam::Vector3 curr_point_;
    gtsam::Vector3 match_point_;
    gtsam::Pose3 Tbl_;

public:
    LidarPose3ICPFactor(gtsam::Key pose_key, gtsam::Pose3 Tbl,
                        gtsam::Vector3 curr_point, gtsam::Vector3 match_point,
                        gtsam::SharedNoiseModel noise_model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise_model, pose_key),
          Tbl_(Tbl),
          curr_point_(curr_point),
          match_point_(match_point) {}

    gtsam::Vector evaluateError(
        const Pose3& pose, boost::optional<Matrix&> H1 = boost::none) const {
        gtsam::Point3 cp(curr_point_(0), curr_point_(1), curr_point_(2));
        gtsam::Point3 mp(match_point_(0), match_point_(1), match_point_(2));

        gtsam::Matrix36 Dpose;
        gtsam::Point3 point_i = Tbl_.transform_from(cp);
        gtsam::Point3 point_w = pose.transform_from(point_i, H1 ? &Dpose : 0);

        gtsam::Vector3 residual;
        residual(0) = point_w(0) - mp(0);
        residual(1) = point_w(1) - mp(1);
        residual(2) = point_w(2) - mp(2);

        // if we need jaccobians
        if (H1) {
            H1->resize(3, 6);

            *H1 << Eigen::Matrix3d::Identity() * Dpose;
        }

        return residual;
    }
};

}  // namespace gtsam