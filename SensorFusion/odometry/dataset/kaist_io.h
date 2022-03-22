#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>

#include "../utils/lidar_utils.h"
#include "type_io.h"

namespace DatasetIO {

void StringSplit(const std::string& str, const std::string& splits,
                 std::vector<std::string>& res) {
    if (str == "")
        return;
    std::string strs = str + splits;
    size_t pos = strs.find(splits);
    int step = splits.size();

    while (pos != strs.npos) {
        std::string temp = strs.substr(0, pos);
        res.push_back(temp);
        strs = strs.substr(pos + step, strs.size());
        pos = strs.find(splits);
    }
}

class KaistIO {
public:
    using Ptr = std::shared_ptr<KaistIO>;
    using PointType = PointVelodyne::Point;
    KaistIO(const std::string& dataset_base_path_)
        : dataset_base_path(dataset_base_path_) {
        std::string timestamp_path =
            dataset_base_path + "/sensor_data/data_stamp.csv";
        load_timestamps(timestamp_path);
        getImuData(imu_data, mag_data);
        getGTData(gt_pose_data);
    }
    ~KaistIO() {}

    std::string dataset_base_path;
    std::unordered_map<std::string, std::vector<long>> sensor_timestamp;
    std::vector<ImuData::Ptr> imu_data;
    std::vector<MagneticData::Ptr> mag_data;
    std::vector<PoseData::Ptr> gt_pose_data;

    StereoImageData::Ptr getStereoData(long timestamp) {
        std::string left_path = dataset_base_path + "/image/stereo_left/" +
                                std::to_string(timestamp) + ".png";
        cv::Mat left_bayer = cv::imread(left_path.c_str(), 0);
        if (left_bayer.empty())
            return nullptr;
        cv::Mat left_rgb;
        cvtColor(left_bayer, left_rgb, CV_BayerRG2BGR);
        std::string right_path = dataset_base_path + "/image/stereo_right/" +
                                 std::to_string(timestamp) + ".png";
        cv::Mat right_bayer = cv::imread(right_path.c_str(), 0);
        if (right_bayer.empty())
            return nullptr;
        cv::Mat right_rgb;
        cvtColor(right_bayer, right_rgb, CV_BayerRG2BGR);

        StereoImageData::Ptr data(new StereoImageData);
        data->timestamp = (double)timestamp * 1e-9;
        data->data[0] = left_rgb.clone();
        data->data[1] = right_rgb.clone();

        return data;
    }

    MonoImageData::Ptr getMonoData(long timestamp) {
        std::string left_path = dataset_base_path + "/image/stereo_left/" +
                                std::to_string(timestamp) + ".png";
        cv::Mat left_bayer = cv::imread(left_path.c_str(), 0);
        if (left_bayer.empty())
            return nullptr;
        cv::Mat left_rgb;
        cvtColor(left_bayer, left_rgb, CV_BayerRG2BGR);

        MonoImageData::Ptr data(new MonoImageData);
        data->timestamp = (double)timestamp * 1e-9;
        data->data = left_rgb.clone();

        return data;
    }

    LidarData<PointType>::Ptr getLeftVlpData(long timestamp) {
        std::string frame_file = dataset_base_path + "/sensor_data/VLP_left/" +
                                 std::to_string(timestamp) + ".bin";
        if (boost::filesystem::exists(frame_file)) {
            std::ifstream file;
            file.open(frame_file, std::ios::in | std::ios::binary);
            LidarData<PointType>::Ptr lidar_data(new LidarData<PointType>);
            lidar_data->data.reset(new pcl::PointCloud<PointType>());
            lidar_data->timestamp = (double)timestamp * 1e-9;
            while (!file.eof()) {
                PointType point;
                file.read(reinterpret_cast<char*>(&point.x), sizeof(float));
                file.read(reinterpret_cast<char*>(&point.y), sizeof(float));
                file.read(reinterpret_cast<char*>(&point.z), sizeof(float));
                file.read(reinterpret_cast<char*>(&point.intensity),
                          sizeof(float));

                float verticalAngle =
                    std::atan2(point.z,
                               sqrt(point.x * point.x + point.y * point.y)) *
                    180 / M_PI;
                point.ring = (verticalAngle + 15.0) / 2.0;
                point.time = -1.0;

                lidar_data->data->points.push_back(point);
            }
            file.close();

            return lidar_data;
        }

        return nullptr;
    }

    LidarData<PointType>::Ptr getRightVlpData(long timestamp) {
        std::string frame_file = dataset_base_path + "/sensor_data/VLP_right/" +
                                 std::to_string(timestamp) + ".bin";
        if (boost::filesystem::exists(frame_file)) {
            std::ifstream file;
            file.open(frame_file, std::ios::in | std::ios::binary);
            LidarData<PointType>::Ptr lidar_data(new LidarData<PointType>);
            lidar_data->data.reset(new pcl::PointCloud<PointType>());
            lidar_data->timestamp = (double)timestamp * 1e-9;
            while (!file.eof()) {
                PointType point;
                file.read(reinterpret_cast<char*>(&point.x), sizeof(float));
                file.read(reinterpret_cast<char*>(&point.y), sizeof(float));
                file.read(reinterpret_cast<char*>(&point.z), sizeof(float));
                file.read(reinterpret_cast<char*>(&point.intensity),
                          sizeof(float));

                float verticalAngle =
                    std::atan2(point.z,
                               sqrt(point.x * point.x + point.y * point.y)) *
                    180 / M_PI;
                point.ring = (verticalAngle + 15.0) / 2.0;

                lidar_data->data->points.push_back(point);
            }
            file.close();

            return lidar_data;
        }

        return nullptr;
    }

    void getImuData(std::vector<ImuData::Ptr>& all_imu_data,
                    std::vector<MagneticData::Ptr>& all_mag_data) {
        std::string imu_data_path =
            dataset_base_path + "/sensor_data/xsens_imu.csv";

        all_imu_data.clear();
        all_mag_data.clear();

        auto s2d = [&](std::string s) {
            std::stringstream ss;
            ss << s;
            double res;
            ss >> res;
            return res;
        };

        std::ifstream in(imu_data_path.c_str());
        if (!in.is_open()) {
            LOG(ERROR) << "kaist getImuData is wrong!";
        }

        while (!in.eof()) {
            std::string s;
            std::getline(in, s);
            if (s[0] == '#' || s == "")
                continue;
            std::vector<std::string> splits;
            StringSplit(s, ",", splits);

            ImuData::Ptr imu_data(new ImuData);
            MagneticData::Ptr mag_data(new MagneticData);

            double stamp = s2d(splits[0]) * 1e-9, q_x = s2d(splits[1]),
                   q_y = s2d(splits[2]), q_z = s2d(splits[3]),
                   q_w = s2d(splits[4]), x = s2d(splits[5]), y = s2d(splits[6]),
                   z = s2d(splits[7]), g_x = s2d(splits[8]),
                   g_y = s2d(splits[9]), g_z = s2d(splits[10]),
                   a_x = s2d(splits[11]), a_y = s2d(splits[12]),
                   a_z = s2d(splits[13]), m_x = s2d(splits[14]),
                   m_y = s2d(splits[15]), m_z = s2d(splits[16]);
            imu_data->timestamp = stamp;
            imu_data->accel(0) = a_x;
            imu_data->accel(1) = a_y;
            imu_data->accel(2) = a_z;
            imu_data->gyro(0) = g_x;
            imu_data->gyro(1) = g_y;
            imu_data->gyro(2) = g_z;
            imu_data->eular_data(0) = x;
            imu_data->eular_data(1) = y;
            imu_data->eular_data(2) = z;
            imu_data->quaternion_data.x() = q_x;
            imu_data->quaternion_data.y() = q_y;
            imu_data->quaternion_data.z() = q_z;
            imu_data->quaternion_data.w() = q_w;
            // std::cout << s << std::endl;
            // std::cout << std::fixed << std::setprecision(8) <<
            // imu_data->accel.transpose()
            //           << std::endl;
            all_imu_data.push_back(imu_data);

            mag_data->timestamp = stamp;
            mag_data->data(0) = m_x;
            mag_data->data(1) = m_y;
            mag_data->data(2) = m_z;
            all_mag_data.push_back(mag_data);
        }

        std::cout << "load " << all_imu_data.size() << " imu sensors' data"
                  << std::endl;
        std::cout << "load " << all_mag_data.size() << " mag sensors' data"
                  << std::endl;
    }

    void getGTData(std::vector<PoseData::Ptr>& all_gt_data) {
        std::string gt_data_path = dataset_base_path + "/global_pose.csv";

        all_gt_data.clear();

        auto s2d = [&](std::string s) {
            std::stringstream ss;
            ss << s;
            double res;
            ss >> res;
            return res;
        };

        std::ifstream in(gt_data_path.c_str());
        if (!in.is_open()) {
            LOG(ERROR) << "kaist getGTData is wrong!";
        }

        while (!in.eof()) {
            std::string s;
            std::getline(in, s);
            if (s[0] == '#' || s == "")
                continue;
            std::vector<std::string> splits;
            StringSplit(s, ",", splits);

            PoseData::Ptr pose_data(new PoseData);

            pose_data->timestamp = s2d(splits[0]) * 1e-9;
            pose_data->data = Eigen::Matrix4d::Identity();
            pose_data->data(0, 0) = s2d(splits[1]);
            pose_data->data(0, 1) = s2d(splits[2]);
            pose_data->data(0, 2) = s2d(splits[3]);
            pose_data->data(0, 3) = s2d(splits[4]);
            pose_data->data(1, 0) = s2d(splits[5]);
            pose_data->data(1, 1) = s2d(splits[6]);
            pose_data->data(1, 2) = s2d(splits[7]);
            pose_data->data(1, 3) = s2d(splits[8]);
            pose_data->data(2, 0) = s2d(splits[9]);
            pose_data->data(2, 1) = s2d(splits[10]);
            pose_data->data(2, 2) = s2d(splits[11]);
            pose_data->data(2, 3) = s2d(splits[12]);
            all_gt_data.push_back(pose_data);
        }
        std::cout << "load " << all_gt_data.size() << " gt data" << std::endl;
    }

private:
    void load_timestamps(const std::string& timestamp_path_) {
        std::ifstream in(timestamp_path_.c_str());
        if (!in.is_open()) {
            LOG(ERROR) << "kaist load_timestamps is wrong!";
        }

        while (!in.eof()) {
            std::string s;
            std::getline(in, s);
            if (s[0] == '#' || s == "")
                continue;

            std::vector<std::string> splits;
            StringSplit(s, ",", splits);

            std::stringstream ss1;
            ss1 << splits[0];
            long ts;
            ss1 >> ts;
            std::string type = splits[1];

            if (sensor_timestamp.find(type) == sensor_timestamp.end()) {
                std::vector<long> timestamps;
                timestamps.push_back(ts);
                sensor_timestamp[type] = timestamps;
            } else {
                sensor_timestamp[type].push_back(ts);
            }
        }

        // Outur Info
        std::cout << std::endl
                  << "Kaist Dataset load successfully !!! " << std::endl;
        std::cout << "load " << sensor_timestamp.size() << " sensors"
                  << std::endl;
        for (auto sensor : sensor_timestamp) {
            std::cout << "    - " << sensor.first << " : "
                      << (int)sensor.second.size() << " data" << std::endl;
            // if (sensor.first == "stereo") {
            //     for (auto ts : sensor.second) {
            //         getStereoData(ts);
            //     }
            // }
        }
        std::cout << std::endl;
    }
};
}  // namespace DatasetIO