#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/lidar_utils.h"

namespace utility {

cv::Mat drawKeypoint(const cv::Mat &img, const std::vector<cv::Point2f> &pts);

cv::Mat drawKeypoint(const cv::Mat &img, const std::vector<cv::KeyPoint> &kpts);

cv::Mat drawKeypoint1(const cv::Mat &img_, const std::vector<cv::Point2f> &pts,
                      const std::vector<int> &cnts = std::vector<int>());

cv::Mat drawKeypoint1(const cv::Mat &img, const std::vector<cv::KeyPoint> &kpts,
                      const std::vector<int> &cnts = std::vector<int>());

cv::Mat drawInlier(
    const cv::Mat &src1, const cv::Mat &src2,
    const std::vector<cv::Point2f> &pt1, const std::vector<cv::Point2f> &pt2,
    const std::vector<cv::DMatch> &inlier = std::vector<cv::DMatch>(),
    int type = 4);

cv::Mat drawInlier(
    const cv::Mat &src1_, const cv::Mat &src2_,
    const std::vector<cv::KeyPoint> &kpt1,
    const std::vector<cv::KeyPoint> &kpt2,
    const std::vector<cv::DMatch> &inlier_ = std::vector<cv::DMatch>(),
    int type = 4);

cv::Mat drawInlier1(const cv::Mat &src1, const cv::Mat &src2,
                    const std::vector<cv::Point2f> &pt1,
                    const std::vector<cv::Point2f> &pt2,
                    const std::vector<uchar> &inlier = std::vector<uchar>(),
                    int type = 4);
cv::Mat drawInlier1(const cv::Mat &src1, const cv::Mat &src2,
                    const std::vector<cv::KeyPoint> &kpt1,
                    const std::vector<cv::KeyPoint> &kpt2,
                    const std::vector<uchar> &inlier = std::vector<uchar>(),
                    int type = 4);

cv::Mat drawTwoImages(const cv::Mat &src1, const cv::Mat &src2, int type = 4);

cv::Mat drawDiffImage(const cv::Mat &src1, const cv::Mat &src2);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertToRGB(
    const SensorFusion::PointCloudType &featureLaserCloud);

}  // namespace utility