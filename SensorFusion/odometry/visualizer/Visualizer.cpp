#include "Visualizer.h"

namespace Visualizer {

Visualizer::Visualizer(const Config& config, const bool& use_thread)
    : config_(config) {
    // Start viz_thread.
    if (use_thread)
        viz_thread_.reset(new std::thread(&Visualizer::Run, this));
}

void Visualizer::DrawCameras(const std::vector<Pose>& camera_poses) {
    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    camera_poses_ = camera_poses;
}

void Visualizer::DrawImuPose(const Pose& pose) {
    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    imu_traj_.emplace_back(pose);
    if (imu_traj_.size() > config_.max_traj_length) {
        imu_traj_.pop_front();
    }
}

void Visualizer::DrawGroundTruth(const std::vector<Pose>& gt_poses) {
    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    double gt_size = gt_poses.size();
    double gap = gt_size / config_.max_traj_length;
    if (gap <= 1)
        gt_traj_ = gt_poses;
    else {
        gt_traj_.clear();
        for (double i = 0; i < gt_size + 0.5;) {
            int idx = (int)i;
            gt_traj_.push_back(gt_poses[idx]);
            i += gap;
        }
    }
}

void Visualizer::DrawGPS(const Pose& pose) {
    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    gps_points_.emplace_back(pose);
    if (gps_points_.size() > config_.max_gps_length) {
        gps_points_.pop_front();
    }
}

void Visualizer::DrawOdom(const Pose& pose) {
    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    odom_traj_.emplace_back(pose);
    if (odom_traj_.size() > config_.max_traj_length) {
        odom_traj_.pop_front();
    }
}

pangolin::OpenGlMatrix SE3ToOpenGlMat(const Pose& pose) {
    pangolin::OpenGlMatrix p_mat;

    auto R = pose.R_;
    auto t = pose.t_;

    p_mat.m[0] = R(0, 0);
    p_mat.m[1] = R(1, 0);
    p_mat.m[2] = R(2, 0);
    p_mat.m[3] = 0.;

    p_mat.m[4] = R(0, 1);
    p_mat.m[5] = R(1, 1);
    p_mat.m[6] = R(2, 1);
    p_mat.m[7] = 0.;

    p_mat.m[8] = R(0, 2);
    p_mat.m[9] = R(1, 2);
    p_mat.m[10] = R(2, 2);
    p_mat.m[11] = 0.;

    p_mat.m[12] = t(0);
    p_mat.m[13] = t(1);
    p_mat.m[14] = t(2);
    p_mat.m[15] = 1.;

    return p_mat;
}

void Visualizer::DrawOneCamera(const Pose& pose) {
    const float w = config_.cam_size;
    const float h = w * 0.75;
    const float z = w * 0.6;

    pangolin::OpenGlMatrix G_T_C = SE3ToOpenGlMat(pose);

    glPushMatrix();

#ifdef HAVE_GLES
    glMultMatrixf(G_T_C.m);
#else
    glMultMatrixd(G_T_C.m);
#endif

    glLineWidth(config_.cam_line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();

    pangolin::glDrawAxis(G_T_C, 0.2);
}

void Visualizer::DrawOrigin() {
    pangolin::OpenGlMatrix origin = SE3ToOpenGlMat(Pose());
    pangolin::glDrawAxis(origin, config_.frame_size);
}

void Visualizer::DrawCameras() {
    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    for (const Pose& cam : camera_poses_) {
        DrawOneCamera(cam);
    }
}

void Visualizer::DrawTraj(const std::vector<Pose>& traj_data) {
    // for (const auto& pose : traj_data) {
    //     DrawImuFrame(pose);
    // }

    glLineWidth(config_.cam_line_width);
    glBegin(GL_LINE_STRIP);

    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    for (const auto& pose : traj_data) {
        const Eigen::Vector3d& t = pose.t_;
        glVertex3f(t[0], t[1], t[2]);
    }

    glEnd();
}

void Visualizer::DrawTraj(const std::deque<Pose>& traj_data) {
    for (const auto& pose : traj_data) {
        DrawImuFrame(pose);
    }

    glLineWidth(2);
    glBegin(GL_LINE_STRIP);

    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    for (const auto& pose : traj_data) {
        const Eigen::Vector3d& t = pose.t_;
        glVertex3f(t[0], t[1], t[2]);
    }

    glEnd();
}

void Visualizer::DrawGpsPoints() {
    glPointSize(config_.gps_point_size);
    glBegin(GL_POINTS);

    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    for (const auto& pt : gps_points_) {
        glVertex3f(pt.t_(0), pt.t_(1), pt.t_(2));
    }

    glEnd();
}

void Visualizer::DrawImuFrame(const Pose& pose) {
    pangolin::OpenGlMatrix G_T_I = SE3ToOpenGlMat(pose);
    pangolin::glDrawAxis(G_T_I, config_.frame_size);
}

void Visualizer::DrawImuFrame() {
    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    if (imu_traj_.empty()) {
        return;
    }
    DrawImuFrame(imu_traj_.back());
}

void Visualizer::Run() {
    pangolin::CreateWindowAndBind("Visualizer", 1920, 1080);
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Menu.
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0,
                                            pangolin::Attach::Pix(175));

    pangolin::Var<bool> menu_follow_cam("menu.Follow Camera", true, true);
    pangolin::Var<int> grid_scale("menu.Grid Size (m)", 100, 1, 500);
    pangolin::Var<bool> show_grid("menu.Show Grid", true, true);
    pangolin::Var<bool> show_map("menu.Show Map", true, true);
    pangolin::Var<bool> show_cam("menu.Show Camera", true, true);
    pangolin::Var<bool> show_traj("menu.Show Traj", true, true);
    pangolin::Var<bool> show_gt_traj("menu.Show GroundTruth", true, true);
    pangolin::Var<bool> show_raw_odom("menu.Show Raw Odom",
                                      config_.show_raw_odom, true);
    pangolin::Var<bool> show_gps_point("menu.Show GPS", config_.show_gps_points,
                                       true);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1920, 1080, config_.view_point_f,
                                   config_.view_point_f, 960, 540, 0.1, 10000),
        pangolin::ModelViewLookAt(config_.view_point_x, config_.view_point_y,
                                  config_.view_point_z, 0, 0, 0, 1, 0, 0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175),
                                           1.0, -1920.0f / 1080.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    // Draw image.
    pangolin::View& d_image =
        pangolin::Display("image")
            .SetBounds(0.0f, 0.5f, 2.0 / 3.0, 1.0f,
                       config_.img_width / config_.img_height)
            .SetLock(pangolin::LockRight, pangolin::LockBottom);
    pangolin::GlTexture image_texture(config_.img_width, config_.img_height,
                                      GL_RGB, true, 0, GL_RGB,
                                      GL_UNSIGNED_BYTE);

    pangolin::OpenGlMatrix G_T_C;
    G_T_C.SetIdentity();

    running_flag_ = true;
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        {
            std::lock_guard<std::mutex> lg(data_buffer_mutex_);
            if (menu_follow_cam && !camera_poses_.empty()) {
                G_T_C = SE3ToOpenGlMat(camera_poses_.back());
                s_cam.Follow(G_T_C);
            }
        }

        DrawOrigin();

        // Draw grid.
        if (show_grid.Get()) {
            glColor3f(0.3f, 0.3f, 0.3f);
            pangolin::glDraw_z0(grid_scale, 1000);
        }

        // Draw Imu traj.
        if (show_traj.Get()) {
            glColor3f(1.0f, 0.0f, 0.0f);
            DrawTraj(imu_traj_);
            // DrawImuFrame();
        }

        // Draw gt Imu traj.
        if (show_gt_traj.Get()) {
            glColor3f(0.5f, 0.5f, 0.0f);
            DrawTraj(gt_traj_);
        }

        // Draw raw odometry.
        if (show_raw_odom.Get()) {
            glColor3f(1.0f, 0.0f, 1.0f);
            DrawTraj(odom_traj_);
        }

        // Draw camera poses.
        if (show_cam.Get()) {
            glColor3f(0.0f, 1.0f, 0.0f);
            DrawCameras();
        }

        // Draw map points.
        if (show_map.Get()) {
            // glColor3f(0.0f, 0.0f, 1.0f);
            // DrawFeatures();
        }

        // Draw Gps points.
        if (show_gps_point.Get()) {
            glColor3f(0.0f, 1.0f, 1.0f);
            // DrawTraj(gt_traj_);
            DrawGpsPoints();
        }

        // Draw image
        {
            std::lock_guard<std::mutex> lg(data_buffer_mutex_);
            if (!image_.empty()) {
                image_texture.Upload(image_.data, GL_RGB, GL_UNSIGNED_BYTE);
                d_image.Activate();
                glColor3f(1.0, 1.0, 1.0);
                image_texture.RenderToViewport();
            }
        }
        pangolin::FinishFrame();
    }
}

void Visualizer::DrawColorImage(const cv::Mat& image) {
    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    cv::resize(image, image_, cv::Size(config_.img_width, config_.img_height),
               0., 0., cv::INTER_NEAREST);
    cv::flip(image_, image_, 0);
}

void Visualizer::DrawImage(const cv::Mat& image,
                           const std::vector<Eigen::Vector2d>& tracked_fts,
                           const std::vector<Eigen::Vector2d>& new_fts) {
    // Covert gray image to color image.
    cv::Mat color_img = image.clone();

    // Draw features on image.
    for (const Eigen::Vector2d& ft : tracked_fts) {
        cv::circle(color_img, cv::Point(ft[0], ft[1]), 5, cv::Scalar(0, 255, 0),
                   -1);
    }
    for (const Eigen::Vector2d& ft : new_fts) {
        cv::circle(color_img, cv::Point(ft[0], ft[1]), 5, cv::Scalar(255, 0, 0),
                   -1);
    }

    std::lock_guard<std::mutex> lg(data_buffer_mutex_);
    cv::resize(color_img, image_,
               cv::Size(config_.img_width, config_.img_height));
    cv::flip(image_, image_, 0);
}

}  // namespace Visualizer