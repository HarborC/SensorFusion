#pragma once

#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "../odom_estimator.h"
#include "estimator.h"

namespace SensorFusion {

class TightOdomEstimator : public OdomEstimator {
public:
    explicit TightOdomEstimator(const OdomEstimatorConfig& config,
                                const Calibration::Ptr& calib)
        : OdomEstimator(config, calib) {
        configure();
        processing_thread.reset(
            new std::thread(&TightOdomEstimator::processingLoop, this));
    }
    ~TightOdomEstimator() { processing_thread->join(); }

    void configure() {
        estimator.reset(new Estimator(2));

        for (int i = 0; i < config.sensor_list.size(); i++) {
            const std::string& sensor = config.sensor_list[i];
            if (sensor == "camera") {
                CameraModule::Ptr module(new CameraModule());
                if (i == 0)
                    module->prime_flag = true;
                for (const auto& cam : calib->camera_calib) {
                    auto cam_id = cam.first;
                    module->ex_pose[cam_id] =
                        calib->extern_params["imu0"][cam_id];
                }
                module->data_queue = &vision_data_queue;
                module->imu_data_queue = &imu_data_queue_vision;

                estimator->modules.push_back(module);
            } else if (sensor == "lidar") {
                LidarModule::Ptr module(new LidarModule());
                if (i == 0)
                    module->prime_flag = true;
                for (const auto& lidar : calib->lidar_calib) {
                    auto lidar_id = lidar.first;
                    module->ex_pose[lidar_id] =
                        calib->extern_params["imu0"][lidar_id];
                }
                module->data_queue = &lidar_data_queue;
                module->imu_data_queue = &imu_data_queue_lidar;

                estimator->modules.push_back(module);
            }
        }
    }

    void processingLoop() {
        std::cout << "Started tight odometry estimator thread " << std::endl;

        auto print_queue_info = [&]() {
            std::cout << "vision_data_queue size : " << vision_data_queue.size()
                      << " | imu_data_queue_vision size : "
                      << imu_data_queue_vision.size()
                      << " | lidar_data_queue size : "
                      << lidar_data_queue.size()
                      << " | imu_data_queue_lidar size : "
                      << imu_data_queue_lidar.size() << std::endl;
        };

        while (true) {
            estimator->Estimate();
            // print_queue_info();
        }

        std::cout << "Finished tight odometry estimator thread " << std::endl;
    }

private:
    std::shared_ptr<std::thread> processing_thread;
    Estimator::Ptr estimator;
};

}  // namespace SensorFusion