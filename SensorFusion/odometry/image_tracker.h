#pragma once

#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Geometry>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <yaml-cpp/yaml.h>

#include "calibration.hpp"
#include "grider_fast.h"
#include "type.h"
#include "utility.h"

namespace SensorFusion {

struct ImageTrackerCamera {
    typedef std::shared_ptr<ImageTrackerCamera> Ptr;
    ImageTrackerCamera(const camodocal::CameraPtr &cam) : camera_ptr(cam) {
        if (camera_ptr == nullptr) {
            std::cout << "camera_ptr == nullptr" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        width = camera_ptr->imageWidth();
        height = camera_ptr->imageHeight();
        focal = std::max(width, height);
        cx = (float)width / 2.0;
        cy = (float)height / 2.0;
    };
    ~ImageTrackerCamera(){};

    Eigen::Vector3f undistort(const cv::Point2f &uv_dist) {
        Eigen::Vector2f ept1;
        ept1 << uv_dist.x, uv_dist.y;
        Eigen::Vector3d p3d;
        camera_ptr->liftSphere(ept1.cast<double>(), p3d);

        return p3d.cast<float>();
    }

    camodocal::CameraPtr camera_ptr;
    float cx, cy, focal;
    int width, height;
};

struct ImageTrackerConfig {
    using Ptr = std::shared_ptr<ImageTrackerConfig>;
    ImageTrackerConfig() {}
    ImageTrackerConfig(const YAML::Node &config) {
        if (config["track_type"])
            track_type = config["track_type"].as<std::string>();
        if (config["skip_frames"])
            skip_frames = config["skip_frames"].as<int>();
        if (config["num_features"])
            num_features = config["num_features"].as<int>();
        if (config["min_num_features"])
            min_num_features = config["min_num_features"].as<int>();
        if (config["use_multi_thread"])
            use_multi_thread = config["use_multi_thread"].as<bool>();
        if (config["fast_threshold"])
            fast_threshold = config["fast_threshold"].as<int>();
        if (config["grid_x"])
            grid_x = config["grid_x"].as<int>();
        if (config["grid_y"])
            grid_y = config["grid_y"].as<int>();
        if (config["min_px_dist"])
            min_px_dist = config["min_px_dist"].as<int>();
        if (config["histogram_method"])
            histogram_method = config["histogram_method"].as<int>();
        if (config["knn_ratio"])
            knn_ratio = config["knn_ratio"].as<double>();
        print();
    }

    void print() {
        std::cout << "ImageTrackerConfig : " << std::endl;
        std::cout << "  - "
                  << "track_type : " << track_type << std::endl;
        std::cout << "  - "
                  << "skip_frames : " << skip_frames << std::endl;
        std::cout << "  - "
                  << "num_features : " << num_features << std::endl;
        std::cout << "  - "
                  << "min_num_features : " << min_num_features << std::endl;
        std::cout << "  - "
                  << "use_multi_thread : " << use_multi_thread << std::endl;
        std::cout << "  - "
                  << "fast_threshold : " << fast_threshold << std::endl;
        std::cout << "  - "
                  << "grid_x : " << grid_x << std::endl;
        std::cout << "  - "
                  << "grid_y : " << grid_y << std::endl;
        std::cout << "  - "
                  << "min_px_dist : " << min_px_dist << std::endl;
        std::cout << "  - "
                  << "histogram_method : " << histogram_method << std::endl;
        std::cout << "  - "
                  << "knn_ratio : " << knn_ratio << std::endl;
    }

    std::string track_type = "klt";
    int skip_frames = 1;
    int num_features = 200;
    int min_num_features = 50;
    bool use_multi_thread = true;
    int fast_threshold = 20;
    int grid_x = 5;
    int grid_y = 5;
    int min_px_dist = 10;
    int histogram_method = 0;  // 0-RAW, 1-HISTOGRAM, 2-CLAHE
    double knn_ratio = 0.85;
};

class ImageTracker {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<ImageTracker>;
    ImageTracker(const ImageTrackerConfig &config,
                 const Calibration::Ptr &calib)
        : frame_counter(0), last_keypoint_id(0), config(config) {
        input_queue.set_capacity(10);

        {
            for (const auto &c : calib->camera_calib) {
                camera_calib[c.first] =
                    ImageTrackerCamera::Ptr(new ImageTrackerCamera(c.second));
                old_pyramid[c.first] = std::vector<cv::Mat>();
                pyramid[c.first] = std::vector<cv::Mat>();
                old_pts[c.first] = std::vector<cv::Point2f>();
                pts[c.first] = std::vector<cv::Point2f>();
                old_ids[c.first] = std::vector<KeypointId>();
                ids[c.first] = std::vector<KeypointId>();
                old_pts_norm[c.first] = std::vector<Eigen::Vector3f>();
                pts_norm[c.first] = std::vector<Eigen::Vector3f>();
            }
            for (const auto &c : calib->camera_masks) {
                masks[c.first] = c.second.clone();
            }
            use_cam_ids = calib->use_cam_ids;
            use_stereo_pairs = calib->camera_stereo;
        }

        processing_thread.reset(
            new std::thread(&ImageTracker::processingLoop, this));
    }

    ~ImageTracker() { processing_thread->join(); }

    void processingLoop() {
        ImageTrackerInput::Ptr input_ptr;

        while (true) {
            if (!input_queue.empty()) {
                input_queue.pop(input_ptr);

                if (!input_ptr.get()) {
                    if (output_queue)
                        output_queue->push(nullptr);
                    break;
                }

                processFrame(input_ptr->timestamp, input_ptr);
            }
        }
    }

    void processFrame(double timestamp, ImageTrackerInput::Ptr &new_img) {
        vaild_cam_ids_vec.clear();
        vaild_stereo_pairs.clear();
        std::unordered_set<std::string> vaild_cam_ids_set;

        for (const auto &cam_id : use_cam_ids) {
            if (new_img->img_data.find(cam_id) != new_img->img_data.end()) {
                vaild_cam_ids_set.insert(cam_id);
                vaild_cam_ids_vec.push_back(cam_id);
            }
        }

        for (const auto &p : use_stereo_pairs) {
            if (vaild_cam_ids_set.find(p.first) != vaild_cam_ids_set.end() &&
                vaild_cam_ids_set.find(p.second) != vaild_cam_ids_set.end() &&
                p.first != p.second) {
                vaild_stereo_pairs.push_back(p);
            }
        }

        if (config.track_type == "klt") {
            trackOpticalFlow(timestamp, new_img);
        } else if (config.track_type == "desc") {
        } else {
            trackOpticalFlow(timestamp, new_img);
        }
    }

    void trackOpticalFlow(double timestamp, ImageTrackerInput::Ptr &new_img) {
        ImageTrackerResult::Ptr re(new ImageTrackerResult());
        re->timestamp = timestamp;
        re->input_images = new_img;
        bool is_track = false;

        // Step Zero : Pyramid Build
        if (config.use_multi_thread && vaild_cam_ids_vec.size() > 1) {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, vaild_cam_ids_vec.size()),
                [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        const std::string &cam_id = vaild_cam_ids_vec[i];
                        std::vector<cv::Mat> new_pyramid;

                        cv::buildOpticalFlowPyramid(
                            imagePreProcess(new_img->img_data[cam_id],
                                            config.histogram_method),
                            new_pyramid, win_size, pyr_levels);
                        pyramid[cam_id] = new_pyramid;
                    }
                });
        } else {
            for (size_t i = 0; i < vaild_cam_ids_vec.size(); ++i) {
                const std::string &cam_id = vaild_cam_ids_vec[i];
                std::vector<cv::Mat> new_pyramid;

                cv::buildOpticalFlowPyramid(
                    imagePreProcess(new_img->img_data[cam_id],
                                    config.histogram_method),
                    new_pyramid, win_size, pyr_levels);
                pyramid[cam_id] = new_pyramid;
            }
        }

        // Step One : Detect Keypoint
        if (config.use_multi_thread && vaild_cam_ids_vec.size() > 1) {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, vaild_cam_ids_vec.size()),
                [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        const std::string &cam_id = vaild_cam_ids_vec[i];
                        detectKeyPointOpticalFlow(cam_id);
                    }
                });
        } else {
            for (size_t i = 0; i < vaild_cam_ids_vec.size(); ++i) {
                const std::string &cam_id = vaild_cam_ids_vec[i];
                detectKeyPointOpticalFlow(cam_id);
            }
        }

        // Step Two : Frame Track
        if (config.use_multi_thread && vaild_cam_ids_vec.size() > 1) {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, vaild_cam_ids_vec.size()),
                [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        const std::string &cam_id = vaild_cam_ids_vec[i];
                        trackFrameOpticalFlow(cam_id);
                    }
                });
        } else {
            for (size_t i = 0; i < vaild_cam_ids_vec.size(); ++i) {
                const std::string &cam_id = vaild_cam_ids_vec[i];
                trackFrameOpticalFlow(cam_id);
            }
        }

        // Step Three : Save Frame Track Result
        // std::unordered_set<KeypointId> pts_id_set;
        for (size_t i = 0; i < vaild_cam_ids_vec.size(); i++) {
            const auto &cam_id = vaild_cam_ids_vec[i];

            for (size_t j = 0; j < ids[cam_id].size(); j++) {
                const KeypointId &kp_id = ids[cam_id][j];
                // if (pts_id_set.find(kp_id) == pts_id_set.end())
                //     pts_id_set.insert(kp_id);

                Feature xyz_uv_velocity;
                xyz_uv_velocity << (pts_norm[cam_id][j])(0),
                    (pts_norm[cam_id][j])(1), (pts_norm[cam_id][j])(2),
                    pts[cam_id][j].x, pts[cam_id][j].y, 0.0, 0.0;

                if (re->observations.find(kp_id) == re->observations.end())
                    re->observations[kp_id] = std::vector<Observation>();

                re->observations[kp_id].push_back(
                    Observation(cam_id, xyz_uv_velocity));

                if (!is_track)
                    is_track = true;
            }
        }

        // Step Four : Stereo Track
        std::vector<std::vector<cv::Point2f>> pts_stereo(
            vaild_stereo_pairs.size());
        std::vector<std::vector<KeypointId>> ids_stereo(
            vaild_stereo_pairs.size());
        std::vector<std::vector<Eigen::Vector3f>> pts_stereo_norm(
            vaild_stereo_pairs.size());
        std::vector<std::vector<uchar>> vaild_verify_stereo(
            vaild_stereo_pairs.size());
        if (config.use_multi_thread && vaild_stereo_pairs.size() > 1) {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, vaild_stereo_pairs.size()),
                [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        const auto &stereo_pair = vaild_stereo_pairs[i];
                        trackStereoOpticalFlow(
                            stereo_pair.first, stereo_pair.second,
                            pts[stereo_pair.first], ids[stereo_pair.first],
                            pts_norm[stereo_pair.first], pts_stereo[i],
                            ids_stereo[i], pts_stereo_norm[i],
                            vaild_verify_stereo[i]);
                    }
                });
        } else {
            for (size_t i = 0; i < vaild_stereo_pairs.size(); ++i) {
                const auto &stereo_pair = vaild_stereo_pairs[i];
                trackStereoOpticalFlow(
                    stereo_pair.first, stereo_pair.second,
                    pts[stereo_pair.first], ids[stereo_pair.first],
                    pts_norm[stereo_pair.first], pts_stereo[i], ids_stereo[i],
                    pts_stereo_norm[i], vaild_verify_stereo[i]);
            }
        }

        // Step Five : Save Stereo Track Result
        for (size_t i = 0; i < vaild_stereo_pairs.size(); i++) {
            const auto &stereo_pair = vaild_stereo_pairs[i];
            const auto &cam_id0 = stereo_pair.first;
            const auto &cam_id1 = stereo_pair.second;
            for (size_t j = 0; j < vaild_verify_stereo[i].size(); j++) {
                if (vaild_verify_stereo[i][j]) {
                    KeypointId kp_id = ids_stereo[i][j];
                    // if (pts_id_set.find(kp_id) == pts_id_set.end()) {
                    //     Feature xyz_uv_velocity;
                    //     xyz_uv_velocity << pts_norm[cam_id0][j](0),
                    //         pts_norm[cam_id0][j](1), pts_norm[cam_id0][j](2),
                    //         pts[cam_id0][j].x, pts[cam_id0][j].y, 0.0, 0.0;

                    //     re->observations[kp_id].emplace_back(cam_id0,
                    //                                          xyz_uv_velocity);
                    // }

                    Eigen::Matrix<float, 7, 1> xyz_uv_velocity;
                    xyz_uv_velocity << (pts_stereo_norm[i][j])(0),
                        (pts_stereo_norm[i][j])(1), (pts_stereo_norm[i][j])(2),
                        pts_stereo[i][j].x, pts_stereo[i][j].y, 0.0, 0.0;

                    re->observations[kp_id].emplace_back(cam_id1,
                                                         xyz_uv_velocity);

                    if (!is_track)
                        is_track = true;
                }
            }
        }

        for (size_t i = 0; i < vaild_cam_ids_vec.size(); ++i) {
            const std::string &cam_id = vaild_cam_ids_vec[i];

            old_pyramid[cam_id] = pyramid[cam_id];
            old_pts[cam_id] = pts[cam_id];
            old_ids[cam_id] = ids[cam_id];
            old_pts_norm[cam_id] = pts_norm[cam_id];
        }

        if (is_track && output_queue &&
            frame_counter % config.skip_frames == 0) {
            output_queue->push(re);
        }

        frame_counter++;
    }

    bool inBorder(const cv::Point2f &pt, int width, int height) {
        const int BORDER_SIZE = 1;
        int img_x = int(pt.x + 0.5);
        int img_y = int(pt.y + 0.5);
        return BORDER_SIZE <= img_x && img_x < width - BORDER_SIZE &&
               BORDER_SIZE <= img_y && img_y < height - BORDER_SIZE;
    }

    template <class T>
    void reduceVector(std::vector<T> &v, const std::vector<uchar> &status) {
        int j = 0;
        for (size_t i = 0; i < v.size(); i++)
            if (status[i])
                v[j++] = v[i];
        v.resize(j);
    }

    cv::Mat imagePreProcess(const cv::Mat &raw_img,
                            const int &histogram_method) const {
        cv::Mat img;
        if (raw_img.channels() == 3) {
            // if(mbRGB)
            //     cvtColor(raw_img,img,cv::COLOR_RGB2GRAY);
            // else
            cvtColor(raw_img, img, cv::COLOR_BGR2GRAY);
        } else if (raw_img.channels() == 4) {
            // if(mbRGB)
            //     cvtColor(mImGray,img,cv::COLOR_RGBA2GRAY);
            // else
            cvtColor(raw_img, img, cv::COLOR_BGRA2GRAY);
        }

        // Histogram equalize
        if (histogram_method == 1) {
            cv::equalizeHist(img, img);
        } else if (histogram_method == 2) {
            double eq_clip_limit = 10.0;
            cv::Size eq_win_size = cv::Size(8, 8);
            cv::Ptr<cv::CLAHE> clahe =
                cv::createCLAHE(eq_clip_limit, eq_win_size);
            clahe->apply(img, img);
        }

        return img.clone();
    }

    void undistortFeature(const std::string &cam_id, const cv::Point2f &pt,
                          Eigen::Vector3f &pt_norm) {
        pt_norm = camera_calib[cam_id]->undistort(pt);
    }

    void undistortFeatures(const std::string &cam_id,
                           const std::vector<cv::Point2f> &pts,
                           std::vector<Eigen::Vector3f> &pts_norm) {
        pts_norm.resize(pts.size());
        for (size_t i = 0; i < pts.size(); i++) {
            undistortFeature(cam_id, pts.at(i), pts_norm.at(i));
        }
    }

    void geometryVerify(const std::string &cam_id0, const std::string &cam_id1,
                        const std::vector<cv::Point2f> &pts0,
                        const std::vector<cv::Point2f> &pts1,
                        const std::vector<Eigen::Vector3f> &pts0_norm,
                        const std::vector<Eigen::Vector3f> &pts1_norm,
                        std::vector<uchar> &vaild_in,
                        std::vector<uchar> &vaild_out) {
        assert(pts0.size() == pts1.size());
        assert(camera_calib.find(cam_id0) != camera_calib.end());
        assert(camera_calib.find(cam_id1) != camera_calib.end());

        if (pts0.size() < 10) {
            vaild_out.resize(pts0.size(), (uchar)0);
            return;
        }

        vaild_out = vaild_in;

        std::vector<cv::Point2f> pts0_norm_, pts1_norm_;
        std::vector<size_t> idx;
        for (size_t i = 0; i < pts0.size(); i++) {
            if (vaild_in[i]) {
                cv::Point2f pt0_norm(pts0_norm[i](0) / pts0_norm[i](2),
                                     pts0_norm[i](1) / pts0_norm[i](2));
                cv::Point2f pt1_norm(pts1_norm[i](0) / pts1_norm[i](2),
                                     pts1_norm[i](1) / pts1_norm[i](2));
                pts0_norm_.push_back(pt0_norm);
                pts1_norm_.push_back(pt1_norm);
                idx.push_back(i);
            }
        }

        std::vector<uchar> vaild_ransac;
        double max_focal_img0 = camera_calib[cam_id0]->focal;
        double max_focal_img1 = camera_calib[cam_id1]->focal;
        double max_focal = std::max(max_focal_img0, max_focal_img1);
        cv::findFundamentalMat(pts0_norm_, pts1_norm_, cv::FM_RANSAC,
                               1.0 / max_focal, 0.999, vaild_ransac);

        for (size_t i = 0; i < vaild_ransac.size(); i++)
            if (!vaild_ransac[i])
                vaild_out[idx[i]] = (uchar)0;
    }

    void maskFilter(const std::string &cam_id0,
                    const std::vector<cv::Point2f> &pts0,
                    std::vector<uchar> &vaild_out) {
        vaild_out.resize(pts0.size(), (uchar)0);
        if (camera_calib.find(cam_id0) != camera_calib.end()) {
            for (size_t i = 0; i < pts0.size(); i++) {
                if ((int)masks[cam_id0].at<uint8_t>((int)pts0.at(i).y,
                                                    (int)pts0.at(i).x) < 127)
                    continue;

                vaild_out[i] = 1;
            }
        }
    }

    void detectionKeyPointByGrid(const std::vector<cv::Mat> &img_pyr,
                                 const std::string &cam_id, const cv::Mat &mask,
                                 std::vector<cv::Point2f> &pts,
                                 std::vector<KeypointId> &ids, int min_px_dist,
                                 int num_features, int min_num_features,
                                 int grid_x, int grid_y, int threshold) {
        cv::Size size((int)((float)img_pyr.at(0).cols / (float)min_px_dist),
                      (int)((float)img_pyr.at(0).rows / (float)min_px_dist));
        std::vector<uchar> status(pts.size(), (uchar)1);
        cv::Mat grid_2d = cv::Mat::zeros(size, CV_8UC1);
        for (int i = 0; i < pts.size(); i++) {
            const auto &pt = pts[i];
            int x = (int)pt.x, y = (int)pt.y;
            int x_grid = (int)(pt.x / (float)min_px_dist);
            int y_grid = (int)(pt.y / (float)min_px_dist);
            if (x_grid < 0 || x_grid >= size.width || y_grid < 0 ||
                y_grid >= size.height ||
                !inBorder(pt, camera_calib[cam_id]->width,
                          camera_calib[cam_id]->height)) {
                status[i] = uchar(0);
                continue;
            }

            if (grid_2d.at<uint8_t>(y_grid, x_grid) > 127) {
                status[i] = uchar(0);
                continue;
            }

            if (mask.at<uint8_t>(y, x) < 127) {
                status[i] = uchar(0);
                continue;
            }

            grid_2d.at<uint8_t>(y_grid, x_grid) = 255;
        }

        reduceVector(pts, status);
        reduceVector(ids, status);

        int num_feats_added = num_features - (int)pts.size();
        // if (num_feats_added <
        //     std::min(min_num_features, (int)(0.2 * num_features)))
        // return;
        if (num_feats_added <= 0)
            return;

        std::vector<cv::KeyPoint> kpts_added;
        ov_core::Grider_FAST::perform_griding(img_pyr.at(0), mask, kpts_added,
                                              num_feats_added, grid_x, grid_y,
                                              threshold, true);

        std::vector<cv::Point2f> pts_new;
        for (auto &kpt : kpts_added) {
            int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
            int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
            if (x_grid < 0 || x_grid >= size.width || y_grid < 0 ||
                y_grid >= size.height)
                continue;

            if (grid_2d.at<uint8_t>(y_grid, x_grid) > 127)
                continue;

            pts_new.push_back(kpt.pt);
            grid_2d.at<uint8_t>(y_grid, x_grid) = 255;
        }

        for (size_t i = 0; i < pts_new.size(); i++) {
            pts.push_back(pts_new.at(i));
            KeypointId temp = ++last_keypoint_id;
            ids.push_back(temp);
        }
    }

    void matchFrameOpticalFlow(const std::vector<cv::Mat> &last_img,
                               const std::vector<cv::Mat> &curr_img,
                               const std::string &last_cam_id,
                               const std::string &curr_cam_id,
                               std::vector<cv::Point2f> &last_pts,
                               std::vector<cv::Point2f> &curr_pts,
                               std::vector<uchar> &vaild_out) {
        assert(!curr_pts.size() ||
               (curr_pts.size() && last_pts.size() == curr_pts.size()));

        vaild_out.resize(last_pts.size(), (uchar)0);

        if (last_pts.size() < 10) {
            return;
        }

        std::vector<uchar> vaild_klt;
        std::vector<float> error_klt;
        cv::TermCriteria term_crit = cv::TermCriteria(
            cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
        if (curr_pts.size())
            cv::calcOpticalFlowPyrLK(last_img, curr_img, last_pts, curr_pts,
                                     vaild_klt, error_klt, win_size, pyr_levels,
                                     term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
        else
            cv::calcOpticalFlowPyrLK(last_img, curr_img, last_pts, curr_pts,
                                     vaild_klt, error_klt, win_size,
                                     pyr_levels);

        for (size_t i = 0; i < vaild_klt.size(); i++)
            if (vaild_klt[i] &&
                inBorder(curr_pts[i], camera_calib[curr_cam_id]->width,
                         camera_calib[curr_cam_id]->height))
                vaild_out[i] = (uchar)1;
    }

    void detectKeyPointOpticalFlow(const std::string &cam_id) {
        if (old_pyramid[cam_id].empty()) {
            detectionKeyPointByGrid(pyramid[cam_id], cam_id, masks[cam_id],
                                    pts[cam_id], ids[cam_id],
                                    config.min_px_dist, config.num_features,
                                    config.min_num_features, config.grid_x,
                                    config.grid_y, config.fast_threshold);
            undistortFeatures(cam_id, pts[cam_id], pts_norm[cam_id]);
        } else {
            detectionKeyPointByGrid(old_pyramid[cam_id], cam_id, masks[cam_id],
                                    old_pts[cam_id], old_ids[cam_id],
                                    config.min_px_dist, config.num_features,
                                    config.min_num_features, config.grid_x,
                                    config.grid_y, config.fast_threshold);
            undistortFeatures(cam_id, old_pts[cam_id], old_pts_norm[cam_id]);
        }
    }

    void trackFrameOpticalFlow(const std::string &cam_id) {
        if (old_pyramid[cam_id].empty()) {
            return;
        } else {
            std::vector<uchar> vaild_match;
            pts[cam_id] = old_pts[cam_id];
            ids[cam_id] = old_ids[cam_id];

            matchFrameOpticalFlow(old_pyramid[cam_id], pyramid[cam_id], cam_id,
                                  cam_id, old_pts[cam_id], pts[cam_id],
                                  vaild_match);

            std::vector<uchar> vaild_mask;
            maskFilter(cam_id, pts[cam_id], vaild_mask);
            for (size_t i = 0; i < vaild_mask.size(); i++)
                if (vaild_mask[i] && !vaild_match[i])
                    vaild_mask[i] = (uchar)0;

            undistortFeatures(cam_id, pts[cam_id], pts_norm[cam_id]);

            std::vector<uchar> vaild_verify;
            geometryVerify(cam_id, cam_id, old_pts[cam_id], pts[cam_id],
                           old_pts_norm[cam_id], pts_norm[cam_id], vaild_mask,
                           vaild_verify);

            reduceVector(pts[cam_id], vaild_verify);
            reduceVector(ids[cam_id], vaild_verify);
            reduceVector(pts_norm[cam_id], vaild_verify);
        }
    }

    void trackStereoOpticalFlow(
        const std::string &cam_id0, const std::string &cam_id1,
        std::vector<cv::Point2f> &pts0, std::vector<KeypointId> &ids0,
        std::vector<Eigen::Vector3f> &pts0_norm, std::vector<cv::Point2f> &pts1,
        std::vector<KeypointId> &ids1, std::vector<Eigen::Vector3f> &pts1_norm,
        std::vector<uchar> &vaild_verify) {
        std::vector<uchar> vaild_match;

        pts1 = pts0;
        ids1 = ids0;

        matchFrameOpticalFlow(pyramid[cam_id0], pyramid[cam_id1], cam_id0,
                              cam_id1, pts0, pts1, vaild_match);

        std::vector<uchar> vaild_mask;
        maskFilter(cam_id1, pts1, vaild_mask);
        for (size_t i = 0; i < vaild_mask.size(); i++)
            if (vaild_mask[i] && !vaild_match[i])
                vaild_mask[i] = (uchar)0;

        undistortFeatures(cam_id1, pts1, pts1_norm);

        geometryVerify(cam_id0, cam_id1, pts0, pts1, pts0_norm, pts1_norm,
                       vaild_mask, vaild_verify);
    }

    tbb::concurrent_bounded_queue<ImageTrackerInput::Ptr> input_queue;
    tbb::concurrent_bounded_queue<ImageTrackerResult::Ptr> *output_queue =
        nullptr;

private:
    std::shared_ptr<std::thread> processing_thread;

    size_t frame_counter;
    std::atomic<KeypointId> last_keypoint_id;

    ImageTrackerConfig config;
    std::unordered_map<std::string, ImageTrackerCamera::Ptr> camera_calib;
    std::unordered_map<std::string, cv::Mat> masks;

    std::vector<std::string> use_cam_ids;
    std::vector<std::pair<std::string, std::string>> use_stereo_pairs;
    std::vector<std::string> vaild_cam_ids_vec;
    std::vector<std::pair<std::string, std::string>> vaild_stereo_pairs;

    // OpticalFlow
    int pyr_levels = 3;
    cv::Size win_size = cv::Size(21, 21);
    std::unordered_map<std::string, std::vector<cv::Mat>> old_pyramid, pyramid;
    std::unordered_map<std::string, std::vector<cv::Point2f>> old_pts, pts;
    std::unordered_map<std::string, std::vector<KeypointId>> old_ids, ids;
    std::unordered_map<std::string, std::vector<Eigen::Vector3f>> old_pts_norm,
        pts_norm;
};

}  // namespace SensorFusion
