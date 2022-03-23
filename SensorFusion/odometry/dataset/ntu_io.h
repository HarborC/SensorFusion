#pragma once

#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>

#define private public
#include <rosbag/bag.h>
#include <rosbag/view.h>
#undef private

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <boost/filesystem.hpp>

#include "../utils/lidar_utils.h"
#include "type_io.h"
#include "utils.h"

namespace DatasetIO {

class NtuIO {
public:
    using Ptr = std::shared_ptr<NtuIO>;
    using PointType = PointOuster::Point;
    NtuIO(const std::string &dataset_base_path_, const std::string &bag_name_)
        : bag_name(bag_name_),
          dataset_base_path(dataset_base_path_),
          bag_path(dataset_base_path + "/" + bag_name + "/" + bag_name +
                   ".bag") {
        load_bags();
        // getGTData(gt_pose_data);
    }
    ~NtuIO() {}

    std::string dataset_base_path;
    std::string bag_name;
    std::string bag_path;
    std::unordered_map<std::string, std::vector<long>> sensor_timestamp;
    std::unordered_map<
        std::string, std::unordered_map<
                         long, std::vector<std::optional<rosbag::IndexEntry>>>>
        sensor_data_idx;
    std::vector<ImuData::Ptr> imu_data;
    std::vector<MagneticData::Ptr> mag_data;
    std::vector<PoseData::Ptr> gt_pose_data;

    std::shared_ptr<rosbag::Bag> bag;
    std::mutex m;

    StereoImageData::Ptr getStereoData(int idx) {
        StereoImageData::Ptr data(new StereoImageData);

        double ts = 0.0;
        for (int i = 0; i < 2; i++) {
            std::string id = (i == 0) ? "stereo_left" : "stereo_right";
            long timestamp = sensor_timestamp[id][idx];
            auto it = sensor_data_idx[id].find(timestamp);
            if (it == sensor_data_idx[id].end() || !it->second[0].has_value())
                return nullptr;

            ts += timestamp;

            m.lock();
            sensor_msgs::ImageConstPtr img_msg =
                bag->instantiateBuffer<sensor_msgs::Image>(*it->second[0]);
            m.unlock();

            cv_bridge::CvImageConstPtr ptr;
            if (img_msg->encoding == "8UC1") {
                sensor_msgs::Image img;
                img.header = img_msg->header;
                img.height = img_msg->height;
                img.width = img_msg->width;
                img.is_bigendian = img_msg->is_bigendian;
                img.step = img_msg->step;
                img.data = img_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img,
                                          sensor_msgs::image_encodings::MONO8);
            } else
                ptr = cv_bridge::toCvCopy(img_msg,
                                          sensor_msgs::image_encodings::MONO8);

            data->data[i] = ptr->image.clone();
        }

        data->timestamp = ts / 2.0 * 1e-9;

        return data;
    }

    MonoImageData::Ptr getMonoData(int idx) {
        long timestamp = sensor_timestamp["stereo_left"][idx];
        auto it = sensor_data_idx["stereo_left"].find(timestamp);
        if (it != sensor_data_idx["stereo_left"].end() &&
            it->second[0].has_value()) {
            m.lock();
            sensor_msgs::ImageConstPtr img_msg =
                bag->instantiateBuffer<sensor_msgs::Image>(*it->second[0]);
            m.unlock();

            cv_bridge::CvImageConstPtr ptr;
            if (img_msg->encoding == "8UC1") {
                sensor_msgs::Image img;
                img.header = img_msg->header;
                img.height = img_msg->height;
                img.width = img_msg->width;
                img.is_bigendian = img_msg->is_bigendian;
                img.step = img_msg->step;
                img.data = img_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img,
                                          sensor_msgs::image_encodings::MONO8);
            } else
                ptr = cv_bridge::toCvCopy(img_msg,
                                          sensor_msgs::image_encodings::MONO8);

            MonoImageData::Ptr data(new MonoImageData);
            data->timestamp = (double)timestamp * 1e-9;
            data->data = ptr->image.clone();

            return data;
        }

        return nullptr;
    }

    LidarData<PointType>::Ptr getHorizOusterData(long timestamp) {
        auto it = sensor_data_idx["horizon_ouster"].find(timestamp);
        if (it != sensor_data_idx["horizon_ouster"].end() &&
            it->second[0].has_value()) {
            LidarData<PointType>::Ptr lidar_data(new LidarData<PointType>);
            lidar_data->data.reset(new pcl::PointCloud<PointType>());
            lidar_data->timestamp = (double)timestamp * 1e-9;

            m.lock();
            sensor_msgs::PointCloud2ConstPtr msg =
                bag->instantiateBuffer<sensor_msgs::PointCloud2>(
                    *it->second[0]);
            m.unlock();

            pcl::PointCloud<PointOuster::Point> laserCloudOrig;
            pcl::fromROSMsg(*msg, *(lidar_data->data));

            return lidar_data;
        }

        return nullptr;
    }

    LidarData<PointType>::Ptr getVertiOusterData(long timestamp) {
        auto it = sensor_data_idx["vertical_ouster"].find(timestamp);
        if (it != sensor_data_idx["vertical_ouster"].end() &&
            it->second[0].has_value()) {
            LidarData<PointType>::Ptr lidar_data(new LidarData<PointType>);
            lidar_data->data.reset(new pcl::PointCloud<PointType>());
            lidar_data->timestamp = (double)timestamp * 1e-9;

            m.lock();
            sensor_msgs::PointCloud2ConstPtr msg =
                bag->instantiateBuffer<sensor_msgs::PointCloud2>(
                    *it->second[0]);
            m.unlock();

            pcl::PointCloud<PointOuster::Point> laserCloudOrig;
            pcl::fromROSMsg(*msg, *(lidar_data->data));

            return lidar_data;
        }

        return nullptr;
    }

    void getGTData(std::vector<PoseData::Ptr> &all_gt_data) {
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
            LOG(ERROR) << "ntu getGTData is wrong!";
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
    void load_bags() {
        if (!boost::filesystem::exists(bag_path)) {
            std::cerr << "No dataset found in " << bag_path << std::endl;
            std::exit(0);
        }

        bag.reset(new rosbag::Bag);
        bag->open(bag_path, rosbag::bagmode::Read);

        rosbag::View view(*bag);

        // get topics
        std::string cam_left_topic = "/left/image_raw";
        std::string cam_right_topic = "/right/image_raw";
        std::string lidar_horz_topic = "/os1_cloud_node1/points";
        std::string lidar_vert_topic = "/os1_cloud_node2/points";
        std::string imu_topic = "/imu/imu";

        sensor_data_idx["stereo_left"] = std::unordered_map<
            long, std::vector<std::optional<rosbag::IndexEntry>>>();
        sensor_data_idx["stereo_right"] = std::unordered_map<
            long, std::vector<std::optional<rosbag::IndexEntry>>>();
        sensor_data_idx["horizon_ouster"] = std::unordered_map<
            long, std::vector<std::optional<rosbag::IndexEntry>>>();
        sensor_data_idx["vertical_ouster"] = std::unordered_map<
            long, std::vector<std::optional<rosbag::IndexEntry>>>();
        sensor_timestamp["stereo_left"] = std::vector<long>();
        sensor_timestamp["stereo_right"] = std::vector<long>();
        sensor_timestamp["horizon_ouster"] = std::vector<long>();
        sensor_timestamp["vertical_ouster"] = std::vector<long>();
        sensor_timestamp["xsens_imu"] = std::vector<long>();

        int num_msgs = 0;

        double min_time = std::numeric_limits<double>::max();
        double max_time = std::numeric_limits<double>::min();

        for (const rosbag::MessageInstance &m : view) {
            const std::string &topic = m.getTopic();

            if (topic == imu_topic) {
                sensor_msgs::ImuConstPtr imu_msg =
                    m.instantiate<sensor_msgs::Imu>();
                long time = imu_msg->header.stamp.toNSec();

                sensor_timestamp["xsens_imu"].push_back(time);

                ImuData::Ptr imu_tm(new ImuData());
                imu_tm->timestamp = (double)time * 1e-9;
                imu_tm->accel = Eigen::Vector3d(imu_msg->linear_acceleration.x,
                                                imu_msg->linear_acceleration.y,
                                                imu_msg->linear_acceleration.z);
                imu_tm->gyro = Eigen::Vector3d(imu_msg->angular_velocity.x,
                                               imu_msg->angular_velocity.y,
                                               imu_msg->angular_velocity.z);

                imu_data.push_back(imu_tm);
            } else if (topic == cam_left_topic) {
                sensor_msgs::ImageConstPtr img_msg =
                    m.instantiate<sensor_msgs::Image>();
                long timestamp_ns = img_msg->header.stamp.toNSec();

                auto &img_vec = sensor_data_idx["stereo_left"][timestamp_ns];
                if (img_vec.size() == 0) {
                    img_vec.resize(1);
                }
                img_vec[0] = m.index_entry_;
                sensor_timestamp["stereo_left"].push_back(timestamp_ns);
            } else if (topic == cam_right_topic) {
                sensor_msgs::ImageConstPtr img_msg =
                    m.instantiate<sensor_msgs::Image>();
                long timestamp_ns = img_msg->header.stamp.toNSec();

                auto &img_vec = sensor_data_idx["stereo_right"][timestamp_ns];
                if (img_vec.size() == 0) {
                    img_vec.resize(1);
                }
                img_vec[0] = m.index_entry_;
                sensor_timestamp["stereo_right"].push_back(timestamp_ns);
            } else if (topic == lidar_horz_topic) {
                sensor_msgs::PointCloud2ConstPtr lidar_msg =
                    m.instantiate<sensor_msgs::PointCloud2>();
                long timestamp_ns = lidar_msg->header.stamp.toNSec();

                auto &img_vec = sensor_data_idx["horizon_ouster"][timestamp_ns];
                if (img_vec.size() == 0)
                    img_vec.resize(1);
                img_vec[0] = m.index_entry_;

                sensor_timestamp["horizon_ouster"].push_back(timestamp_ns);
            } else if (topic == lidar_vert_topic) {
                sensor_msgs::PointCloud2ConstPtr lidar_msg =
                    m.instantiate<sensor_msgs::PointCloud2>();
                long timestamp_ns = lidar_msg->header.stamp.toNSec();

                auto &img_vec =
                    sensor_data_idx["vertical_ouster"][timestamp_ns];
                if (img_vec.size() == 0)
                    img_vec.resize(1);
                img_vec[0] = m.index_entry_;

                sensor_timestamp["vertical_ouster"].push_back(timestamp_ns);
            }

            num_msgs++;
        }

        // Outur Info
        std::cout << std::endl
                  << "Ntu Dataset load successfully !!! " << std::endl;
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