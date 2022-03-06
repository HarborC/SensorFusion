#include "odom_estimator.h"
// #include "estimator/loose_odom_estimator.h"
#include "estimator/tight_odom_estimator.h"

namespace SensorFusion {

OdomEstimator::Ptr OdomEstimatorFactory::getOdomEstimator(
    const OdomEstimatorConfig &config, const Calibration::Ptr &calib) {
    OdomEstimator::Ptr res;

    if (config.odom_type == "loose") {
        // res.reset(new LooseOdomEstimator(config, calib));
    } else if (config.odom_type == "tight") {
        res.reset(new TightOdomEstimator(config, calib));
    }

    return res;
}

}  // namespace SensorFusion