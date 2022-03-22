/*
 * @Author: Jiagang Chen
 * @Date: 2021-12-09 17:41:59
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-12-10 01:21:55
 * @Description: ...
 * @Reference: ...
 */
#ifndef _VISUALIZER_VISUALIZER_H_
#define _VISUALIZER_VISUALIZER_H_

#include "Common.h"
#include "Pose.h"

namespace Visualizer {

class Visualizer {
public:
    struct Config {
        double cam_size = 2;
        double cam_line_width = 3.;
        double point_size = 2.;
        double frame_size = 2.;
        double view_point_x = 0.;
        double view_point_y = 0.;
        double view_point_z = 200.;
        double view_point_f = 500.;

        double img_height = 250;
        double img_width = 500;

        int max_traj_length = 20000;
        int max_num_features = 5000;

        int max_gps_length = 10000;
        double gps_point_size = 5.;

        bool show_raw_odom = false;
        bool show_gps_points = true;
    };

    typedef std::shared_ptr<Visualizer> Ptr;
    Visualizer(const Config& config, const bool& use_thread = true);
    ~Visualizer() {
        if (viz_thread_)
            viz_thread_->join();
    }

    void Run();

    void DrawCameras(const std::vector<Pose>& camera_poses);

    void DrawImuPose(const Pose& pose);

    void DrawGPS(const Pose& pose);

    void DrawGroundTruth(const std::vector<Pose>& gt_poses);

    void DrawOdom(const Pose& pose);

    void DrawColorImage(const cv::Mat& image);
    void DrawImage(const cv::Mat& image,
                   const std::vector<Eigen::Vector2d>& tracked_fts,
                   const std::vector<Eigen::Vector2d>& new_fts);

private:
    void DrawOrigin();
    void DrawOneCamera(const Pose& pose);
    void DrawCameras();
    void DrawTraj(const std::deque<Pose>& traj_data);
    void DrawTraj(const std::vector<Pose>& traj_data);
    void DrawImuFrame(const Pose& pose);
    void DrawImuFrame();
    void DrawGpsPoints();

    const Config config_;

    // Thread.
    std::shared_ptr<std::thread> viz_thread_;
    std::atomic<bool> running_flag_;

    // Data buffer.
    std::mutex data_buffer_mutex_;
    std::vector<Pose> camera_poses_;
    std::deque<Pose> imu_traj_;
    std::vector<Pose> gt_traj_;
    std::deque<Pose> odom_traj_;
    std::deque<Pose> gps_points_;

    cv::Mat image_;
};

}  // namespace Visualizer

#endif