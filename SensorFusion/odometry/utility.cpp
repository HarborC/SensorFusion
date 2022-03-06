#include "utility.h"

namespace utility {

cv::Mat drawKeypoint(const cv::Mat &img, const std::vector<cv::Point2f> &pts) {
    std::vector<cv::KeyPoint> kpts;
    cv::KeyPoint::convert(pts, kpts, 1, 1, 0, -1);
    return drawKeypoint(img, kpts);
}

cv::Mat drawKeypoint(const cv::Mat &img,
                     const std::vector<cv::KeyPoint> &kpts) {
    cv::Mat output;
    cv::drawKeypoints(img, kpts, output);
    return output.clone();
}

cv::Mat drawKeypoint1(const cv::Mat &img_, const std::vector<cv::Point2f> &pts,
                      const std::vector<int> &cnts) {
    if (cnts.size())
        assert(cnts.size() == pts.size());

    cv::Mat imTrack = img_.clone();
    if (img_.channels() != 3) {
        cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);
    }

    for (size_t j = 0; j < pts.size(); j++) {
        double len = 1.0;
        if (cnts.size() > 0)
            len = std::min(1.0, 1.0 * cnts[j] / 20);
        cv::circle(imTrack, pts[j], 2,
                   cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    return imTrack.clone();
}

cv::Mat drawKeypoint1(const cv::Mat &img, const std::vector<cv::KeyPoint> &kpts,
                      const std::vector<int> &cnts) {
    std::vector<cv::Point2f> pts;
    cv::KeyPoint::convert(kpts, pts);
    return drawKeypoint1(img, pts, cnts);
}

cv::Mat drawInlier(const cv::Mat &src1, const cv::Mat &src2,
                   const std::vector<cv::Point2f> &pt1,
                   const std::vector<cv::Point2f> &pt2,
                   const std::vector<cv::DMatch> &inlier, int type) {
    std::vector<cv::KeyPoint> kpt1, kpt2;
    cv::KeyPoint::convert(pt1, kpt1, 1, 1, 0, -1);
    cv::KeyPoint::convert(pt2, kpt2, 1, 1, 0, -1);

    return drawInlier(src1, src2, kpt1, kpt2, inlier, type);
}

cv::Mat drawInlier(const cv::Mat &src1_, const cv::Mat &src2_,
                   const std::vector<cv::KeyPoint> &kpt1,
                   const std::vector<cv::KeyPoint> &kpt2,
                   const std::vector<cv::DMatch> &inlier_, int type) {
    cv::Mat src1 = src1_.clone(), src2 = src2_.clone();
    if (src1_.channels() != 3) {
        cv::cvtColor(src1_, src1, CV_GRAY2RGB);
    }
    if (src2_.channels() != 3) {
        cv::cvtColor(src2_, src2, CV_GRAY2RGB);
    }

    std::vector<cv::DMatch> inlier = inlier_;
    if (inlier_.empty()) {
        for (int i = 0; i < kpt1.size(); i++) {
            cv::DMatch m;
            m.queryIdx = i;
            m.trainIdx = i;
            m.distance = 100;
            inlier.push_back(m);
        }
    }

    cv::Mat output;
    if (type / 3 == 0) {
        int height = std::max(src1.rows, src2.rows);
        int width = src1.cols + src2.cols;
        output = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

        if (type % 3 != 2) {
            src1.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
            src2.copyTo(output(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));
        } else {
            cv::Mat src1_n, src2_n;
            cv::drawKeypoints(src1, kpt1, src1_n);
            cv::drawKeypoints(src2, kpt2, src2_n);
            src1_n.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
            src2_n.copyTo(output(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));
        }
    } else {
        int height = src1.rows + src2.rows;
        int width = std::max(src1.cols, src2.cols);
        output = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

        if (type % 3 != 2) {
            src1.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
            src2.copyTo(output(cv::Rect(0, src1.rows, src2.cols, src2.rows)));
        } else {
            cv::Mat src1_n, src2_n;
            cv::drawKeypoints(src1, kpt1, src1_n);
            cv::drawKeypoints(src2, kpt2, src2_n);
            src1_n.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
            src2_n.copyTo(output(cv::Rect(0, src1.rows, src2.cols, src2.rows)));
        }
    }

    if (type == 0 || type == 2) {
        for (size_t i = 0; i < inlier.size(); i++) {
            cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
            cv::Point2f right = (kpt2[inlier[i].trainIdx].pt +
                                 cv::Point2f((float)src1.cols, 0.f));
            cv::line(output, left, right, cv::Scalar(0, 255, 255));
        }
    } else if (type == 1) {
        for (size_t i = 0; i < inlier.size(); i++) {
            cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
            cv::Point2f right = (kpt2[inlier[i].trainIdx].pt +
                                 cv::Point2f((float)src1.cols, 0.f));
            cv::line(output, left, right, cv::Scalar(255, 0, 0));
        }

        for (size_t i = 0; i < inlier.size(); i++) {
            cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
            cv::Point2f right = (kpt2[inlier[i].trainIdx].pt +
                                 cv::Point2f((float)src1.cols, 0.f));
            cv::circle(output, left, 1, cv::Scalar(0, 255, 255), 2);
            cv::circle(output, right, 1, cv::Scalar(0, 255, 0), 2);
        }
    } else if (type == 3 || type == 5) {
        for (size_t i = 0; i < inlier.size(); i++) {
            cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
            cv::Point2f right = (kpt2[inlier[i].trainIdx].pt +
                                 cv::Point2f(0.f, (float)src1.rows));
            cv::line(output, left, right, cv::Scalar(0, 255, 255));
        }
    } else if (type == 4) {
        for (size_t i = 0; i < inlier.size(); i++) {
            cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
            cv::Point2f right = (kpt2[inlier[i].trainIdx].pt +
                                 cv::Point2f(0.f, (float)src1.rows));
            cv::line(output, left, right, cv::Scalar(255, 0, 0));
        }

        for (size_t i = 0; i < inlier.size(); i++) {
            cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
            cv::Point2f right = (kpt2[inlier[i].trainIdx].pt +
                                 cv::Point2f(0.f, (float)src1.rows));
            cv::circle(output, left, 1, cv::Scalar(0, 255, 255), 2);
            cv::circle(output, right, 1, cv::Scalar(0, 255, 0), 2);
        }
    }

    return output;
}

cv::Mat drawInlier1(const cv::Mat &src1, const cv::Mat &src2,
                    const std::vector<cv::Point2f> &pt1,
                    const std::vector<cv::Point2f> &pt2,
                    const std::vector<uchar> &inlier, int type) {
    std::vector<cv::KeyPoint> kpt1, kpt2;
    cv::KeyPoint::convert(pt1, kpt1, 1, 1, 0, -1);
    cv::KeyPoint::convert(pt2, kpt2, 1, 1, 0, -1);
    return drawInlier1(src1, src2, kpt1, kpt2, inlier, type);
}

cv::Mat drawInlier1(const cv::Mat &src1, const cv::Mat &src2,
                    const std::vector<cv::KeyPoint> &kpt1,
                    const std::vector<cv::KeyPoint> &kpt2,
                    const std::vector<uchar> &inlier, int type) {
    std::vector<cv::DMatch> match;
    if (inlier.empty()) {
        for (int i = 0; i < kpt1.size(); i++) {
            cv::DMatch m;
            m.queryIdx = i;
            m.trainIdx = i;
            m.distance = 100;
            match.push_back(m);
        }
    } else {
        for (int i = 0; i < inlier.size(); i++) {
            if (inlier[i]) {
                cv::DMatch m;
                m.queryIdx = i;
                m.trainIdx = i;
                m.distance = 100;
                match.push_back(m);
            }
        }
    }

    return drawInlier(src1, src2, kpt1, kpt2, match, type);
}

cv::Mat drawTwoImages(const cv::Mat &src1, const cv::Mat &src2, int type) {
    std::vector<cv::KeyPoint> kpt1, kpt2;
    std::vector<cv::DMatch> match;

    return drawInlier(src1, src2, kpt1, kpt2, match, type);
}

cv::Mat drawDiffImage(const cv::Mat &src1_, const cv::Mat &src2_) {
    assert(src1_.size() == src2_.size());

    cv::Mat src1 = src1_.clone(), src2 = src2_.clone();
    if (src1_.channels() != 1) {
        cv::cvtColor(src1_, src1, CV_RGB2GRAY);
    }

    if (src2_.channels() != 1) {
        cv::cvtColor(src2_, src2, CV_RGB2GRAY);
    }

    cv::Size size(src1.cols, src1.rows);
    cv::Mat diff = cv::Mat::zeros(size, CV_8UC1);

    for (int i = 0; i < src1.rows; i++) {
        for (int j = 0; j < src1.cols; j++) {
            diff.at<uchar>(i, j) = (uchar)(std::abs((int)src1.at<uchar>(i, j) -
                                                    (int)src2.at<uchar>(i, j)));
        }
    }

    return diff.clone();
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertToRGB(
    const SensorFusion::PointCloudType &featureLaserCloud) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr viewPoints(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    for (const auto &p : featureLaserCloud.points) {
        pcl::PointXYZRGB point;
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;
        if (std::fabs(p.normal_z - 1.0) < 1e-5) {
            // Red
            point.r = 255;
            point.g = 0;
            point.b = 0;
            viewPoints->points.push_back(point);
        } else if (std::fabs(p.normal_z - 2.0) < 1e-5) {
            // Green
            point.r = 0;
            point.g = 255;
            point.b = 0;
            viewPoints->points.push_back(point);
        }
    }

    return viewPoints;
}

}  // namespace utility