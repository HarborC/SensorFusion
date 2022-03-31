#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam {

class LidarPose3EdgeFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
private:
    gtsam::Vector3 curr_point_, last_point_a_, last_point_b_;
    gtsam::Pose3 Tbl_;

public:
    LidarPose3EdgeFactor(gtsam::Key pose_key, gtsam::Pose3 Tbl,
                         gtsam::Vector3 curr_point, gtsam::Vector3 last_point_a,
                         gtsam::Vector3 last_point_b,
                         gtsam::SharedNoiseModel noise_model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise_model, pose_key),
          Tbl_(Tbl),
          curr_point_(curr_point),
          last_point_a_(last_point_a),
          last_point_b_(last_point_b) {}

    gtsam::Vector evaluateError(
        const Pose3& pose, boost::optional<Matrix&> H1 = boost::none) const {
        gtsam::Point3 cp(curr_point_(0), curr_point_(1), curr_point_(2));
        gtsam::Point3 lpa(last_point_a_(0), last_point_a_(1), last_point_a_(2));
        gtsam::Point3 lpb(last_point_b_(0), last_point_b_(1), last_point_b_(2));

        gtsam::Matrix36 Dpose;
        gtsam::Point3 pi = Tbl_.transform_from(cp);
        gtsam::Point3 pw = pose.transform_from(pi, H1 ? &Dpose : 0);

        gtsam::Point3 nu = (pw - lpa).cross(pw - lpb);
        gtsam::Point3 de = lpa - lpb;
        double de_norm = de.norm();

        gtsam::Vector3 residual;
        residual(0) = nu.x() / de_norm;
        residual(1) = nu.y() / de_norm;
        residual(2) = nu.z() / de_norm;

        // if we need jaccobians
        if (H1) {
            H1->resize(3, 6);

            *H1 << -(gtsam::skewSymmetric(de.x(), de.y(), de.z()) * Dpose) /
                       de.norm();
        }

        return residual;
    }
};
}  // namespace gtsam