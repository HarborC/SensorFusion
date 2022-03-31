#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

#include <fmt/format.h>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>

#include <CLI/CLI.hpp>

#include "odometry/calibration.hpp"
#include "odometry/global.h"
#include "odometry/image_tracker.h"
#include "odometry/lidar_extractor.h"
#include "odometry/odom_config.h"
#include "odometry/odom_estimator.h"
#include "odometry/visualizer/Visualizer.h"

#include "odometry/dataset/ntu_io.h"

using namespace SensorFusion;

using namespace fmt::literals;

std::mutex m;
std::condition_variable con_var;
bool step_by_step = false;
size_t max_frames = 0;

// all variables
Calibration::Ptr calibration;
DatasetIO::NtuIO::Ptr ntu_dataset_io;
OdometryConfig::Ptr odom_config;
ImageTracker::Ptr image_tracker;
OdomEstimator::Ptr odom_estimator;
LidarExtractor::Ptr lidar_extractor;

long start_timestamp = 0;
bool need_realtime = true;

tbb::concurrent_bounded_queue<State::Ptr> out_state_queue;

// Feed functions
void feed_images() {
    std::cout << "Started feed_images thread " << std::endl;

    // Image Tracker
    image_tracker.reset(
        new ImageTracker(*(odom_config->image_tracker_config), calibration));

    image_tracker->output_queue = &(odom_estimator->vision_data_queue);

    const std::vector<long>& sensor_timestamp =
        ntu_dataset_io->sensor_timestamp["stereo_left"];
    for (int i = 0; i < (int)sensor_timestamp.size(); i++) {
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();

        if (start_timestamp > 0) {
            if (sensor_timestamp[i] < start_timestamp) {
                continue;
            }
        }

        auto mono_data = ntu_dataset_io->getMonoData(i);
        ImageTrackerInput::Ptr input(new ImageTrackerInput);
        input->timestamp = mono_data->timestamp;
        input->img_data["cam0"] = mono_data->data.clone();

        mono_data->show();

        // image_tracker->input_queue.push(input);

        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();

        if (need_realtime && i != (int)sensor_timestamp.size() - 1) {
            double dt =
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                          t1)
                    .count();

            double T = (sensor_timestamp[i + 1] - sensor_timestamp[i]) * 1e-9;

            if (dt < T)
                usleep((T - dt) * 1e6);
        }
    }

    std::cout << "Finished feed_images thread " << std::endl;
}

void feed_imu() {
    std::cout << "Started feed_imu thread " << std::endl;

    const auto& sensor_data = ntu_dataset_io->imu_data;
    for (int i = 0; i < (int)sensor_data.size(); i++) {
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();

        if (start_timestamp > 0) {
            if (sensor_data[i]->timestamp < start_timestamp) {
                continue;
            }
        }

        ImuData::Ptr imu(new ImuData);
        imu->timeStamp = sensor_data[i]->timestamp;
        imu->accel = sensor_data[i]->accel;
        imu->gyro = sensor_data[i]->gyro;
        odom_estimator->addIMUToQueue(imu);

        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();

        if (need_realtime && i != (int)sensor_data.size() - 1) {
            double dt =
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                          t1)
                    .count();

            double T =
                (sensor_data[i + 1]->timestamp - sensor_data[i]->timestamp);

            if (dt < T)
                usleep((T - dt) * 1e6);
        }
    }

    std::cout << "Finished feed_imu thread " << std::endl;
}

void feed_lidar() {
    std::cout << "Started feed_lidar thread " << std::endl;

    lidar_extractor.reset(
        new LidarExtractor(*(calibration->lidar_calib["lidar0"]), false));

    const std::vector<long>& sensor_timestamp =
        ntu_dataset_io->sensor_timestamp["horizon_ouster"];
    for (int i = 0; i < (int)sensor_timestamp.size(); i++) {
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();

        if (start_timestamp > 0) {
            if (sensor_timestamp[i] < start_timestamp) {
                continue;
            }
        }

        auto lidar_data =
            ntu_dataset_io->getHorizOusterData(sensor_timestamp[i]);
        LidarFeatureResult::Ptr output = lidar_extractor->ousterHandler(
            lidar_data->timestamp, *(lidar_data->data));
        output->sensor_id = "lidar0";
        odom_estimator->lidar_data_queue.push(output);

        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();

        if (need_realtime && i != (int)sensor_timestamp.size() - 1) {
            double dt =
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                          t1)
                    .count();

            double T = (sensor_timestamp[i + 1] - sensor_timestamp[i]) * 1e-9;

            if (dt < T)
                usleep((T - dt) * 1e6);
        }
    }

    std::cout << "Finished feed_lidar thread " << std::endl;
}

int main(int argc, char** argv) {
    bool show_gui = true;
    bool print_queue = false;
    std::string project_path = "/shared/SensorFusion/";
    std::string calib_path = project_path + "config/ntu/sensor_config.yaml";
    std::string dataset_path = "/media/cjg/SLAM/Datasets/NTU/";
    std::string dataset_name = "eee_03";
    std::string config_path = project_path + "config/ntu/odom_config.yaml";
    std::string result_path = project_path + "result/ntu/";
    std::string trajectory_fmt = "tum";
    bool trajectory_groundtruth;
    int num_threads = 8;

    start_timestamp = 0;
    need_realtime = true;

    // CLI
    {
        CLI::App app{"App description"};
        app.add_option("--show-gui", show_gui, "Show GUI");
        app.add_option(
            "--calib", calib_path,
            "Ground-truth calibration used for simulation.");  //->required();
        app.add_option("--dataset-path", dataset_path,
                       "Path to dataset.");  //->required();
        app.add_option("--print-queue", print_queue, "Print queue.");
        app.add_option("--config-path", config_path, "Path to config file.");
        app.add_option(
            "--result-path", result_path,
            "Path to result file where the system will write RMSE ATE.");
        app.add_option("--num-threads", num_threads, "Number of threads.");
        app.add_option("--step-by-step", step_by_step, "Path to config file.");
        app.add_option(
            "--save-trajectory", trajectory_fmt,
            "Save trajectory. Supported formats <tum, euroc, kitti>");
        app.add_option("--save-groundtruth", trajectory_groundtruth,
                       "In addition to trajectory, save also ground turth");
        app.add_option("--max-frames", max_frames,
                       "Limit number of frames to process from dataset (0 "
                       "means unlimited)");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            return app.exit(e);
        }
    }

    // load data
    ntu_dataset_io.reset(new DatasetIO::NtuIO(dataset_path, dataset_name));
    calibration.reset(new Calibration(calib_path));
    odom_config.reset(new OdometryConfig(config_path));
    logger.reset(new Logger(result_path));

    // global thread limit is in effect until global_control object is destroyed
    std::unique_ptr<tbb::global_control> tbb_global_control;
    if (num_threads > 0) {
        tbb_global_control = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, num_threads);
    }

    {
        // Odom Estimator
        odom_estimator = OdomEstimatorFactory::getOdomEstimator(
            *(odom_config->odom_estimator_config), calibration);

        // if (show_gui) {
        //     Visualizer::Visualizer::Config viz_config;
        //     visualizer.reset(new Visualizer::Visualizer(viz_config, false));
        // }
    }

    std::thread t1(&feed_images);
    std::thread t2(&feed_imu);
    std::thread t3(&feed_lidar);

    if (show_gui) {
        std::vector<Visualizer::Pose> gt_pose;
        Eigen::Vector3d gt_base;
        for (int i = 0; i < ntu_dataset_io->gt_pose_data.size(); i++) {
            const auto& gt_p = ntu_dataset_io->gt_pose_data[i];

            if (i == 0) {
                gt_base = gt_p->data.block<3, 1>(0, 3);
            }

            Eigen::Matrix3d R = gt_p->data.block<3, 3>(0, 0);
            Eigen::Vector3d t = gt_p->data.block<3, 1>(0, 3) - gt_base;
            gt_pose.push_back(Visualizer::Pose(gt_p->timestamp, R, t));
        }
        g_viz->DrawGroundTruth(gt_pose);

        g_viz->Run();
    }

    // join input threads
    t1.join();
    t2.join();
    t3.join();

    std::cout << "END-----------" << std::endl;

    return 0;
}
