#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

#include <fmt/format.h>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include "odometry/calibration.hpp"
#include "odometry/image_tracker.h"
#include "odometry/lidar_extractor.h"
#include "odometry/odom_config.h"
#include "odometry/odom_estimator.h"

#include "odometry/dataset/kaist_io.h"

using namespace SensorFusion;

using namespace fmt::literals;

// Pangolin variables
constexpr int UI_WIDTH = 200;

using Button = pangolin::Var<std::function<void(void)>>;

pangolin::DataLog imu_data_log, odom_data_log, error_data_log;
pangolin::Plotter* plotter;

pangolin::Var<int> show_frame("ui.show_frame", 0, 0, 1500);

pangolin::Var<bool> show_flow("ui.show_flow", false, false, true);
pangolin::Var<bool> show_obs("ui.show_obs", true, false, true);
pangolin::Var<bool> show_ids("ui.show_ids", false, false, true);

pangolin::Var<bool> show_est_pos("ui.show_est_pos", true, false, true);
pangolin::Var<bool> show_est_vel("ui.show_est_vel", false, false, true);
pangolin::Var<bool> show_est_bg("ui.show_est_bg", false, false, true);
pangolin::Var<bool> show_est_ba("ui.show_est_ba", false, false, true);

pangolin::Var<bool> show_gt("ui.show_gt", true, false, true);

// Button next_step_btn("ui.next_step", &next_step);
// Button prev_step_btn("ui.prev_step", &prev_step);

pangolin::Var<bool> continue_btn("ui.continue", false, false, true);
pangolin::Var<bool> continue_fast("ui.continue_fast", true, false, true);

// Button align_se3_btn("ui.align_se3", &alignButton);

pangolin::Var<bool> euroc_fmt("ui.euroc_fmt", true, false, true);
pangolin::Var<bool> tum_rgbd_fmt("ui.tum_rgbd_fmt", false, false, true);
pangolin::Var<bool> kitti_fmt("ui.kitti_fmt", false, false, true);
pangolin::Var<bool> save_groundtruth("ui.save_groundtruth", false, false, true);
// Button save_traj_btn("ui.save_traj", &saveTrajectoryButton);

pangolin::Var<bool> follow("ui.follow", true, false, true);

// pangolin::Var<bool> record("ui.record", false, false, true);

pangolin::OpenGlRenderState camera;

// // Visualization variables
// std::unordered_map<int64_t, basalt::VioVisualizationData::Ptr> vis_map;

// tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr>
// out_vis_queue;
// tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr>
//     out_state_queue;

// std::vector<int64_t> odom_t_ns;
// Eigen::aligned_vector<Eigen::Vector3d> odom_t_w_i;
// Eigen::aligned_vector<Sophus::SE3d> odom_T_w_i;

// std::vector<int64_t> gt_t_ns;
// Eigen::aligned_vector<Eigen::Vector3d> gt_t_w_i;

// std::string marg_data_path;
// size_t last_frame_processed = 0;

// tbb::concurrent_unordered_map<int64_t, int, std::hash<int64_t>>
// timestamp_to_id;

std::mutex m;
std::condition_variable con_var;
bool step_by_step = false;
size_t max_frames = 0;

std::atomic<bool> terminate = false;

// all variables
Calibration::Ptr calibration;
DatasetIO::KaistIO::Ptr kaist_dataset_io;
OdometryConfig::Ptr odom_config;
ImageTracker::Ptr image_tracker;
OdomEstimator::Ptr odom_estimator;
LidarExtractor::Ptr lidar_extractor;

tbb::concurrent_bounded_queue<State::Ptr> out_state_queue;

// Feed functions
void feed_images() {
    std::cout << "Started feed_images thread " << std::endl;

    // Image Tracker
    image_tracker.reset(
        new ImageTracker(*(odom_config->image_tracker_config), calibration));

    image_tracker->output_queue = &(odom_estimator->vision_data_queue);

    const std::vector<long>& sensor_timestamp =
        kaist_dataset_io->sensor_timestamp["stereo"];
    for (int i = 0; i < (int)sensor_timestamp.size(); i++) {
        auto mono_data = kaist_dataset_io->getMonoData(sensor_timestamp[i]);
        ImageTrackerInput::Ptr input(new ImageTrackerInput);
        input->timestamp = mono_data->timestamp;
        input->img_data["cam0"] = mono_data->data.clone();

        image_tracker->input_queue.push(input);
    }

    std::cout << "Finished feed_images thread " << std::endl;
}

void feed_imu() {
    std::cout << "Started feed_imu thread " << std::endl;

    const auto& sensor_data = kaist_dataset_io->imu_data;
    for (int i = 0; i < (int)sensor_data.size(); i++) {
        ImuData::Ptr imu(new ImuData);
        imu->timeStamp = sensor_data[i]->timestamp;
        imu->accel = sensor_data[i]->accel;
        imu->gyro = sensor_data[i]->gyro;
        odom_estimator->addIMUToQueue(imu);
    }

    std::cout << "Finished feed_imu thread " << std::endl;
}

void feed_lidar() {
    std::cout << "Started feed_lidar thread " << std::endl;

    lidar_extractor.reset(
        new LidarExtractor(*(calibration->lidar_calib["lidar0"]), false));

    const std::vector<long>& sensor_timestamp =
        kaist_dataset_io->sensor_timestamp["velodyne_left"];
    for (int i = 0; i < (int)sensor_timestamp.size(); i++) {
        auto lidar_data = kaist_dataset_io->getLeftVlpData(sensor_timestamp[i]);
        LidarFeatureResult::Ptr output(new LidarFeatureResult);
        output->timestamp = lidar_data->timestamp;
        output->sensor_id = "lidar0";
        output->features =
            lidar_extractor->velodyneHandler(*(lidar_data->data));
        odom_estimator->lidar_data_queue.push(output);
    }

    std::cout << "Finished feed_lidar thread " << std::endl;
}

int main(int argc, char** argv) {
    bool show_gui = true;
    bool print_queue = false;
    std::string project_path = "/shared/SensorFusion/";
    std::string calib_path = project_path + "config/kaist/sensor_config.yaml";
    std::string dataset_path =
        "/media/cjg/SLAM/Datasets/KAIST/raw/Urban28/urban28-pankyo/";
    std::string dataset_type;
    std::string config_path = project_path + "config/kaist/odom_config.yaml";
    std::string result_path = project_path + "result/";
    std::string trajectory_fmt = "tum";
    bool trajectory_groundtruth;
    int num_threads = 12;

    // CLI
    {
        CLI::App app{"App description"};
        app.add_option("--show-gui", show_gui, "Show GUI");
        app.add_option(
            "--calib", calib_path,
            "Ground-truth calibration used for simulation.");  //->required();
        app.add_option("--dataset-path", dataset_path,
                       "Path to dataset.");  //->required();
        app.add_option("--dataset-type", dataset_type,
                       "Dataset type <euroc, bag>.");  //->required();
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
    kaist_dataset_io.reset(new DatasetIO::KaistIO(dataset_path));
    calibration.reset(new Calibration(calib_path));
    odom_config.reset(new OdometryConfig(config_path));

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

        // // if (show_gui)
        // //     odom_estimator->out_vis_queue = &out_vis_queue;
        // odom_estimator->out_state_queue = &out_state_queue;
    }

    std::thread t1(&feed_images);
    std::thread t2(&feed_imu);
    std::thread t3(&feed_lidar);

    {}

    // join input threads
    t1.join();
    t2.join();
    t3.join();

    // std::cout << "Data input finished, terminate auxiliary threads.";
    terminate = true;

    std::cout << "END-----------" << std::endl;

    return 0;
}
