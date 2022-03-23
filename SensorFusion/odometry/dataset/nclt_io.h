#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>

#include "../utils/lidar_utils.h"
#include "type_io.h"
#include "utils.h"

namespace DatasetIO {

class NcltIO {
public:
    using Ptr = std::shared_ptr<NcltIO>;
    using PointType = PointVelodyne::Point;
    NcltIO(const std::string& root_path_, const std::string& seq_name_)
        : root_path(root_path_),
          seq_name(seq_name_),
          dataset_base_path(root_path + "/" + seq_name + "/") {
        std::string timestamp_path = dataset_base_path + "/data_stamp.csv";
        load_timestamps(timestamp_path);

        // gen_map_bin();
        load_img_maps();

        getImuData(imu_data, mag_data);
        getGTData(gt_pose_data);
    }
    ~NcltIO() {}

    std::string root_path;
    std::string seq_name;
    std::string dataset_base_path;
    std::unordered_map<std::string, std::vector<long>> sensor_timestamp;
    std::unordered_map<std::string, std::pair<cv::Mat, cv::Mat>> img_maps;
    std::vector<ImuData::Ptr> imu_data;
    std::vector<MagneticData::Ptr> mag_data;
    std::vector<PoseData::Ptr> gt_pose_data;

    MultiImageData::Ptr getMultiData(long timestamp) {
        MultiImageData::Ptr data(new MultiImageData);
        data->timestamp = (double)timestamp * 1e-6;
        for (int i = 0; i < 6; i++) {
            std::string rgb_path = dataset_base_path + "/lb3/Cam" +
                                   std::to_string(i) + "/" +
                                   std::to_string(timestamp) + ".tiff";
            cv::Mat rgb = cv::imread(rgb_path.c_str(), 0);
            if (rgb.empty())
                return nullptr;
            data->data.push_back(rgb);
        }

        return data;
    }

    cv::Mat undistort_img(const std::string& cam_id, const cv::Mat& raw_img) {
        cv::Mat undis_img;
        cv::remap(raw_img, undis_img, img_maps[cam_id].first,
                  img_maps[cam_id].second, cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        return undis_img.clone();
    }

    MonoImageData::Ptr getMonoData(long timestamp) {
        std::string rgb_path = dataset_base_path + "/lb3/Cam5/" +
                               std::to_string(timestamp) + ".tiff";
        cv::Mat rgb = cv::imread(rgb_path.c_str(), 0);
        if (rgb.empty())
            return nullptr;

        MonoImageData::Ptr data(new MonoImageData);
        data->timestamp = (double)timestamp * 1e-6;
        data->data = rgb.clone();

        return data;
    }

    MonoImageData::Ptr getPanoData(long timestamp) {
        std::string rgb_path = dataset_base_path + "/lb3/panoImgs/" +
                               std::to_string(timestamp) + ".jpg";
        cv::Mat rgb = cv::imread(rgb_path.c_str(), 0);
        if (rgb.empty())
            return nullptr;

        MonoImageData::Ptr data(new MonoImageData);
        data->timestamp = (double)timestamp * 1e-6;
        data->data = rgb.clone();

        return data;
    }

    LidarData<PointType>::Ptr getVlpData(long timestamp) {
        std::string frame_file = dataset_base_path + "/velodyne_sync/" +
                                 std::to_string(timestamp) + ".bin";

        auto point_convert = [&](float& x, float& y, float& z) {
            float scaling = 0.005;
            float offset = -100.0;

            x = x * scaling + offset;
            y = y * scaling + offset;
            z = z * scaling + offset;
        };

        if (boost::filesystem::exists(frame_file)) {
            std::ifstream file;
            file.open(frame_file, std::ios::in | std::ios::binary);
            LidarData<PointType>::Ptr lidar_data(new LidarData<PointType>);
            lidar_data->data.reset(new pcl::PointCloud<PointType>());
            lidar_data->timestamp = (double)timestamp * 1e-6;
            while (!file.eof()) {
                PointType point;
                short x, y, z;
                file.read(reinterpret_cast<char*>(&x), sizeof(short));
                file.read(reinterpret_cast<char*>(&y), sizeof(short));
                file.read(reinterpret_cast<char*>(&z), sizeof(short));
                point.x = x;
                point.y = y;
                point.z = z;
                point_convert(point.x, point.y, point.z);

                unsigned char intensity, ring;
                file.read(reinterpret_cast<char*>(&intensity),
                          sizeof(unsigned char));
                file.read(reinterpret_cast<char*>(&ring),
                          sizeof(unsigned char));
                point.intensity = intensity;
                point.ring = ring;
                point.time = 1.0;

                lidar_data->data->points.push_back(point);
            }
            file.close();

            return lidar_data;
        }

        return nullptr;
    }

    void getImuData(std::vector<ImuData::Ptr>& all_imu_data,
                    std::vector<MagneticData::Ptr>& all_mag_data) {
        std::string ms25_data_path = dataset_base_path + "/ms25.csv";

        all_imu_data.clear();
        all_mag_data.clear();

        auto s2d = [&](std::string s) {
            std::stringstream ss;
            ss << s;
            double res;
            ss >> res;
            return res;
        };

        std::ifstream in(ms25_data_path.c_str());
        if (!in.is_open()) {
            LOG(ERROR) << "nclt getImuData is wrong!";
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

            double stamp = s2d(splits[0]) * 1e-6, g_x = s2d(splits[7]),
                   g_y = s2d(splits[8]), g_z = s2d(splits[9]),
                   a_x = s2d(splits[4]), a_y = s2d(splits[5]),
                   a_z = s2d(splits[6]), m_x = s2d(splits[1]),
                   m_y = s2d(splits[2]), m_z = s2d(splits[3]);
            imu_data->timestamp = stamp;
            imu_data->accel(0) = a_x;
            imu_data->accel(1) = a_y;
            imu_data->accel(2) = a_z;
            imu_data->gyro(0) = g_x;
            imu_data->gyro(1) = g_y;
            imu_data->gyro(2) = g_z;

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
        std::string gt_data_path = dataset_base_path + "/groundtruth.csv";

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
            LOG(ERROR) << "nclt getGTData is wrong!";
        }

        while (!in.eof()) {
            std::string s;
            std::getline(in, s);
            if (s[0] == '#' || s == "")
                continue;
            std::vector<std::string> splits;
            StringSplit(s, ",", splits);

            PoseData::Ptr pose_data(new PoseData);

            double lon = -s2d(splits[1]), lat = s2d(splits[2]),
                   hei = s2d(splits[3]), roll = s2d(splits[4]),
                   pitch = s2d(splits[5]), yaw = s2d(splits[6]);

            double x = lon * 20037508.342789 / 180;
            double y =
                std::log(std::tan((90 + lat) * M_PI / 360)) / (M_PI / 180);
            y = y * 20037508.34789 / 180;

            Eigen::Matrix3d euler_pose = Eigen::Matrix3d::Identity();

            pose_data->timestamp = s2d(splits[0]) * 1e-6;
            pose_data->data = Eigen::Matrix4d::Identity();
            pose_data->data.block<3, 3>(0, 0) =
                getRotationMatrixFromEulerAnglesRollPitchYaw(roll, pitch, yaw);
            pose_data->data(0, 3) = x;
            pose_data->data(1, 3) = y;
            pose_data->data(2, 3) = hei;
            all_gt_data.push_back(pose_data);
        }
        std::cout << "load " << all_gt_data.size() << " gt data" << std::endl;
    }

private:
    void gen_map_bin() {
        for (int i = 0; i < 6; i++) {
            int H = 1232;
            int W = 1616;

            std::string in_path = root_path +
                                  "/params/U2D_ALL_1616X1232/U2D_Cam" +
                                  std::to_string(i) + "_1616X1232.txt";

            std::ifstream in1(in_path);
            if (!in1.is_open()) {
                cout << "Can't open " << in_path << endl;
                exit(0);
            }

            cout << "Reading " << in_path << endl;

            string s;
            getline(in1, s);

            std::string out_path = root_path +
                                   "/params/U2D_ALL_1616X1232/U2D_Cam" +
                                   std::to_string(i) + "_1616X1232.bin";

            std::ofstream outF(out_path, std::ios::binary);

            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    getline(in1, s);
                    stringstream ss;
                    ss << s;
                    int v1, v2;
                    float v3, v4;
                    ss >> v1;
                    ss >> v2;
                    if (v1 != y || v2 != x) {
                        cout << "Wrong in the LoadMap4Ladybug func..." << endl;
                        exit(0);
                    }

                    ss >> v3;
                    ss >> v4;

                    outF.write(reinterpret_cast<char*>(&v1), sizeof(int));
                    outF.write(reinterpret_cast<char*>(&v2), sizeof(int));
                    outF.write(reinterpret_cast<char*>(&v3), sizeof(float));
                    outF.write(reinterpret_cast<char*>(&v4), sizeof(float));
                }
            }

            in1.close();
            outF.close();
        }
    }

    void load_img_maps() {
        for (int i = 0; i < 6; i++) {
            std::string cam_id = "cam" + std::to_string(i);
            cv::Mat maps_x, maps_y;
            int H = 1232;
            int W = 1616;

            std::string path = root_path + "/params/U2D_ALL_1616X1232/U2D_Cam" +
                               std::to_string(i) + "_1616X1232.bin";

            std::ifstream in1(path, std::ios::binary);
            if (!in1.is_open()) {
                cout << "Can't open " << path << endl;
                exit(0);
            }

            cout << "Reading " << path << endl;

            maps_x.create(H, W, CV_32FC1);
            maps_y.create(H, W, CV_32FC1);
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int v1, v2;
                    in1.read(reinterpret_cast<char*>(&v1), sizeof(int));
                    in1.read(reinterpret_cast<char*>(&v2), sizeof(int));

                    if (v1 != y || v2 != x) {
                        cout << "Wrong in the LoadMap4Ladybug func..." << endl;
                        exit(0);
                    }

                    float v3, v4;
                    in1.read(reinterpret_cast<char*>(&v3), sizeof(float));
                    in1.read(reinterpret_cast<char*>(&v4), sizeof(float));

                    maps_x.at<float>(y, x) = v4;
                    maps_y.at<float>(y, x) = v3;
                }
            }

            in1.close();

            img_maps[cam_id] = std::pair<cv::Mat, cv::Mat>(maps_x, maps_y);
        }
    }

    void load_timestamps(const std::string& timestamp_path_) {
        std::ifstream in(timestamp_path_.c_str());
        if (!in.is_open()) {
            LOG(ERROR) << "nclt load_timestamps is wrong!";
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
                  << "nclt Dataset load successfully !!! " << std::endl;
        std::cout << "load " << sensor_timestamp.size() << " sensors"
                  << std::endl;
        for (auto sensor : sensor_timestamp) {
            std::cout << "    - " << sensor.first << " : "
                      << (int)sensor.second.size() << " data" << std::endl;
        }
        std::cout << std::endl;
    }

    Eigen::Matrix3d getRotationMatrixFromEulerAnglesRollPitchYaw(
        const double& roll_rad, const double& pitch_rad,
        const double& yaw_rad) {
        Eigen::Matrix3d R_yaw;
        R_yaw << std::cos(yaw_rad), -std::sin(yaw_rad), 0.0, std::sin(yaw_rad),
            std::cos(yaw_rad), 0.0, 0.0, 0.0, 1.0;

        Eigen::Matrix3d R_pitch;
        R_pitch << std::cos(pitch_rad), 0.0, std::sin(pitch_rad), 0.0, 1.0, 0.0,
            -std::sin(pitch_rad), 0.0, std::cos(pitch_rad);

        Eigen::Matrix3d R_roll;
        R_roll << 1.0, 0.0, 0.0, 0.0, std::cos(roll_rad), -std::sin(roll_rad),
            0.0, std::sin(roll_rad), std::cos(roll_rad);

        return R_yaw * R_pitch * R_roll;
    }
};
}  // namespace DatasetIO