#include "imu.h"

namespace SensorFusion {

inline ImuData::Ptr interpolate_data(const ImuData::Ptr &imu_1,
                                     const ImuData::Ptr &imu_2,
                                     double timestamp) {
    double lambda =
        (timestamp - imu_1->timestamp) / (imu_2->timestamp - imu_1->timestamp);

    auto nam = (1 - lambda) * imu_1->am + lambda * imu_2->am;
    auto nwm = (1 - lambda) * imu_1->wm + lambda * imu_2->wm;

    ImuData::Ptr data(new ImuData());
    data->timestamp = timestamp;
    data->am = nam;
    data->wm = nwm;

    return data;
}

std::vector<ImuData::Ptr> select_imu_readings(
    const std::deque<ImuData::Ptr> &imu_data, double time0, double time1,
    bool warn) {
    // Our vector imu readings
    std::vector<ImuData::Ptr> prop_data;

    // Ensure we have some measurements in the first place!
    if (imu_data.empty()) {
        if (warn)
            LOG(WARNING)
                << "No IMU measurements. IMU-CAMERA are likely messed up!!!";
        return prop_data;
    }

    // Loop through and find all the needed measurements to propagate with
    // Note we split measurements based on the given state time, and the update
    // timestamp
    for (size_t i = 0; i < imu_data.size() - 1; i++) {
        // START OF THE INTEGRATION PERIOD
        // If the next timestamp is greater then our current state time
        // And the current is not greater then it yet...
        // Then we should "split" our current IMU measurement
        if (imu_data.at(i + 1)->timestamp > time0 &&
            imu_data.at(i)->timestamp < time0) {
            ImuData::Ptr data =
                interpolate_data(imu_data.at(i), imu_data.at(i + 1), time0);
            prop_data.push_back(data);
            // printf("propagation #%d = CASE 1 = %.3f => %.3f\n",
            // (int)i,data.timestamp-prop_data.at(0).timestamp,time0-prop_data.at(0).timestamp);
            continue;
        }

        // MIDDLE OF INTEGRATION PERIOD
        // If our imu measurement is right in the middle of our propagation
        // period Then we should just append the whole measurement time to our
        // propagation vector
        if (imu_data.at(i)->timestamp >= time0 &&
            imu_data.at(i + 1)->timestamp <= time1) {
            prop_data.push_back(imu_data.at(i));
            // printf("propagation #%d = CASE 2 =
            // %.3f\n",(int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp);
            continue;
        }

        // END OF THE INTEGRATION PERIOD
        // If the current timestamp is greater then our update time
        // We should just "split" the NEXT IMU measurement to the update time,
        // NOTE: we add the current time, and then the time at the end of the
        // interval (so we can get a dt) NOTE: we also break out of this loop,
        // as this is the last IMU measurement we need!
        if (imu_data.at(i + 1)->timestamp > time1) {
            // If we have a very low frequency IMU then, we could have only
            // recorded the first integration (i.e. case 1) and nothing else In
            // this case, both the current IMU measurement and the next is
            // greater than the desired intepolation, thus we should just cut
            // the current at the desired time Else, we have hit CASE2 and this
            // IMU measurement is not past the desired propagation time, thus
            // add the whole IMU reading
            if (imu_data.at(i)->timestamp > time1 && i == 0) {
                // This case can happen if we don't have any imu data that has
                // occured before the startup time This means that either we
                // have dropped IMU data, or we have not gotten enough. In this
                // case we can't propgate forward in time, so there is not that
                // much we can do.
                break;
            } else if (imu_data.at(i)->timestamp > time1) {
                ImuData::Ptr data =
                    interpolate_data(imu_data.at(i - 1), imu_data.at(i), time1);
                prop_data.push_back(data);
                // printf("propagation #%d = CASE 3.1 = %.3f => %.3f\n",
                // (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
            } else {
                prop_data.push_back(imu_data.at(i));
                // printf("propagation #%d = CASE 3.2 = %.3f => %.3f\n",
                // (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
            }
            // If the added IMU message doesn't end exactly at the camera time
            // Then we need to add another one that is right at the ending time
            if (prop_data.at(prop_data.size() - 1)->timestamp != time1) {
                ImuData::Ptr data =
                    interpolate_data(imu_data.at(i), imu_data.at(i + 1), time1);
                prop_data.push_back(data);
                // printf("propagation #%d = CASE 3.3 = %.3f => %.3f\n",
                // (int)i,data.timestamp-prop_data.at(0).timestamp,data.timestamp-time0);
            }
            break;
        }
    }

    // Check that we have at least one measurement to propagate with
    if (prop_data.empty()) {
        if (warn)
            LOG(WARNING) << "No IMU measurements to propagate with ("
                         << (int)prop_data.size()
                         << " of 2). IMU-CAMERA are likely messed up!!!";
        return prop_data;
    }

    // If we did not reach the whole integration period (i.e., the last inertial
    // measurement we have is smaller then the time we want to reach) Then we
    // should just "stretch" the last measurement to be the whole period (case 3
    // in the above loop) if(time1-imu_data.at(imu_data.size()-1).timestamp >
    // 1e-3) {
    //    printf(YELLOW "ImuPropagator::select_imu_readings(): Missing inertial
    //    measurements to propagate with (%.6f sec missing). IMU-CAMERA are
    //    likely messed up!!!\n" RESET,
    //    (time1-imu_data.at(imu_data.size()-1).timestamp)); return prop_data;
    //}

    // Loop through and ensure we do not have an zero dt values
    // This would cause the noise covariance to be Infinity
    for (size_t i = 0; i < prop_data.size() - 1; i++) {
        if (std::abs(prop_data.at(i + 1)->timestamp -
                     prop_data.at(i)->timestamp) < 1e-12) {
            if (warn)
                LOG(WARNING) << "Zero DT between IMU reading" << (int)i
                             << " and " << (int)(i + 1) << ", removing it!";
            prop_data.erase(prop_data.begin() + i);
            i--;
        }
    }

    // Check that we have at least one measurement to propagate with
    if (prop_data.size() < 2) {
        if (warn)
            LOG(WARNING) << "No IMU measurements to propagate with ("
                         << (int)prop_data.size()
                         << " of 2). IMU-CAMERA are likely messed up!!!";
        return prop_data;
    }

    // Success :D
    return prop_data;
}

}  // namespace SensorFusion
