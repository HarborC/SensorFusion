#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

// yaml-cpp
#include <yaml-cpp/yaml.h>

// glog
#include <glog/logging.h>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace DatasetIO {

enum class MeasureType {
    kUnknown = 0,
    kMonoImage = 1,
    kSimMonoImage = 2,
    kGnss = 3,
    kWheel = 4,
    kGyro = 5,
    kAccel = 6,
    kPose = 7,
    kStereoImage = 8,
    kMultiImage = 9,
    kImu = 10,
    kLidar = 11,
    kMagnetic = 12
};

struct Measurement {
    typedef std::shared_ptr<Measurement> Ptr;
    virtual ~Measurement() {}

    double timestamp;
    MeasureType type;
};

struct MonoImageData : public Measurement {
    typedef std::shared_ptr<MonoImageData> Ptr;
    MonoImageData() : exposure(0) { type = MeasureType::kMonoImage; }
    ~MonoImageData() override = default;

    void show(double last_timestamp = -1) {
        if (!data.empty()) {
            cv::imshow("WS_left", data);
            if (last_timestamp == -1) {
                cv::waitKey(10);
            } else {
                cv::waitKey((timestamp - last_timestamp) * 500);
            }
        }
    }

    cv::Mat data;
    double exposure;
};

struct StereoImageData : public Measurement {
    typedef std::shared_ptr<StereoImageData> Ptr;
    StereoImageData() { type = MeasureType::kStereoImage; }
    ~StereoImageData() override = default;
    void show(double last_timestamp = -1) {
        if (!data[0].empty()) {
            cv::imshow("WS_left", data[0]);
            if (last_timestamp == -1) {
                cv::waitKey(15);
            } else {
                cv::waitKey((timestamp - last_timestamp) * 500);
            }
        }

        if (!data[1].empty()) {
            cv::imshow("WS_right", data[1]);
            if (last_timestamp == -1) {
                cv::waitKey(15);
            } else {
                cv::waitKey((timestamp - last_timestamp) * 500);
            }
        }
    }

    cv::Mat data[2];
    double exposure[2] = {0, 0};
};

struct MultiImageData : public Measurement {
    typedef std::shared_ptr<MultiImageData> Ptr;
    MultiImageData() { type = MeasureType::kMultiImage; }
    ~MultiImageData() override = default;

    std::vector<cv::Mat> data;
    std::vector<double> exposure;
};

struct GyroData : public Measurement {
    typedef std::shared_ptr<GyroData> Ptr;
    GyroData() { type = MeasureType::kGyro; }
    ~GyroData() override = default;

    Eigen::Vector3d data;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct AccelData : public Measurement {
    typedef std::shared_ptr<AccelData> Ptr;
    AccelData() { type = MeasureType::kAccel; }
    ~AccelData() override = default;

    Eigen::Vector3d data;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PoseData : public Measurement {
    typedef std::shared_ptr<PoseData> Ptr;
    PoseData() { type = MeasureType::kPose; }
    ~PoseData() override = default;

    Eigen::Matrix4d data;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct WheelData : public Measurement {
    typedef std::shared_ptr<WheelData> Ptr;
    WheelData() { type = MeasureType::kWheel; }
    ~WheelData() override = default;

    Eigen::Vector2d data;  // left and right
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct GnssData : public Measurement {
    typedef std::shared_ptr<GnssData> Ptr;
    GnssData() { type = MeasureType::kGnss; }
    ~GnssData() override = default;

    Eigen::Vector3d data;  // lat, lon, height
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ImuData : public Measurement {
    typedef std::shared_ptr<ImuData> Ptr;
    ImuData() { type = MeasureType::kImu; }
    ~ImuData() override = default;

    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
    Eigen::Vector3d eular_data;
    Eigen::Quaterniond quaternion_data;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct MagneticData : public Measurement {
    typedef std::shared_ptr<MagneticData> Ptr;
    MagneticData() { type = MeasureType::kMagnetic; }
    ~MagneticData() override = default;

    Eigen::Vector3d data;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <class POINTTYPE>
struct LidarData : public Measurement {
    typedef std::shared_ptr<LidarData> Ptr;
    LidarData() { type = MeasureType::kLidar; }
    ~LidarData() override = default;

    typename pcl::PointCloud<POINTTYPE>::Ptr data;
};

}  // namespace DatasetIO
