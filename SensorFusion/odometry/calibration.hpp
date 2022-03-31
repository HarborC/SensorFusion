#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "camera_model/Camera.h"
#include "camera_model/CameraFactory.h"
#include "type.h"
#include "utils/lidar_utils.h"

namespace SensorFusion {

class ImuCalibration {
public:
    using Ptr = std::shared_ptr<ImuCalibration>;
    explicit ImuCalibration(const std::string &calib_config) {
        auto config_node = YAML::LoadFile(calib_config);
        if (config_node["acc_n"])
            acc_n = config_node["acc_n"].as<double>();
        if (config_node["gyr_n"])
            gyr_n = config_node["gyr_n"].as<double>();
        if (config_node["acc_w"])
            acc_w = config_node["acc_w"].as<double>();
        if (config_node["gyr_w"])
            gyr_w = config_node["gyr_w"].as<double>();
        if (config_node["g_norm"])
            g_norm = config_node["g_norm"].as<double>();
    }
    ~ImuCalibration() {}

    double acc_n = 0.08;
    double gyr_n = 0.004;
    double acc_w = 2.0e-4;
    double gyr_w = 2.0e-5;
    double g_norm = 9.805;
};

class LidarCalibration {
public:
    using Ptr = std::shared_ptr<LidarCalibration>;
    explicit LidarCalibration(const std::string &calib_config) {
        auto config_node = YAML::LoadFile(calib_config);
        if (config_node["lidar_type"]) {
            auto type = config_node["lidar_type"].as<std::string>();
            if (type == "velodyne") {
                lidar_type = LidarType::VELODYNE;
            } else if (type == "ouster") {
                lidar_type = LidarType::OUSTER;
            } else if (type == "livox_horizon") {
                lidar_type = LidarType::LIVOX_HORIZON;
            } else if (type == "livox_mid") {
                lidar_type = LidarType::LIVOX_MID;
            } else if (type == "livox_avia") {
                lidar_type = LidarType::LIVOX_AVIA;
            }
        }

        if (config_node["scan_num"])
            scan_num = config_node["scan_num"].as<int>();
        if (config_node["scan_rate"])
            scan_rate = config_node["scan_rate"].as<int>();
        if (config_node["feature_enabled"])
            feature_enabled = config_node["feature_enabled"].as<bool>();
        if (config_node["cov_enabled"])
            cov_enabled = config_node["cov_enabled"].as<bool>();
        if (config_node["point_filter_num"])
            point_filter_num = config_node["point_filter_num"].as<int>();
        if (config_node["blind"])
            blind = config_node["blind"].as<double>();
    }
    ~LidarCalibration() {}

    void print() {
        std::cout << "LidarCalibration : " << std::endl;
        std::cout << "  - "
                  << "lidar_type : " << lidar_type << std::endl;
        std::cout << "  - "
                  << "scan_num : " << scan_num << std::endl;
        std::cout << "  - "
                  << "scan_rate : " << scan_rate << std::endl;
        std::cout << "  - "
                  << "feature_enabled : " << (int)feature_enabled << std::endl;
        std::cout << "  - "
                  << "cov_enabled : " << (int)cov_enabled << std::endl;
        std::cout << "  - "
                  << "point_filter_num : " << point_filter_num << std::endl;
        std::cout << "  - "
                  << "blind : " << blind << std::endl;
    }

    int lidar_type = 0;
    int scan_num = 16;
    int scan_rate = 10;
    bool feature_enabled = true;
    bool cov_enabled = false;
    int point_filter_num = 1;
    double blind = 2.0;
};

class Calibration {
public:
    using Ptr = std::shared_ptr<Calibration>;
    explicit Calibration(const std::string &calib_config) {
        extern_params["body"] = std::map<std::string, Eigen::Matrix4d>();
        load_config(calib_config);
    }
    ~Calibration() {}

    std::map<std::string, ImuCalibration::Ptr> imu_calib;
    std::map<std::string, LidarCalibration::Ptr> lidar_calib;
    std::map<CameraId, camodocal::CameraPtr> camera_calib;
    std::map<std::string, cv::Mat> camera_masks;
    std::vector<std::string> use_cam_ids;
    std::vector<std::pair<std::string, std::string>> camera_stereo;

    std::map<std::string, std::map<std::string, Eigen::Matrix4d>> extern_params;
    std::map<CameraId, Eigen::Matrix4d> Tic;

private:
    void load_config(const std::string &calib_config) {
        auto config_node = YAML::LoadFile(calib_config);
        cv::FileStorage cv_params(calib_config, cv::FileStorage::READ);
        if (config_node["imu"]) {
            int imu_num = config_node["imu"].as<int>();
            for (int i = 0; i < imu_num; i++) {
                std::string id = "imu" + std::to_string(i);
                std::string id_name = id + "_calib";
                if (config_node[id_name]) {
                    ImuCalibration::Ptr imu_cal(new ImuCalibration(
                        config_node[id_name].as<std::string>()));
                    imu_calib[id] = imu_cal;
                } else {
                    LOG(ERROR) << "no config_node[id_name]//imu";
                    std::exit(0);
                }

                std::string ex_name = "T_body_" + id;
                if (config_node[ex_name]) {
                    cv::Mat cv_T;
                    cv_params[ex_name.c_str()] >> cv_T;
                    Eigen::Matrix4d T_body_sensor;
                    cv::cv2eigen(cv_T, T_body_sensor);
                    addNewExternParam(id, T_body_sensor);
                } else {
                    LOG(ERROR) << "no config_node[ex_name]//imu";
                    std::exit(0);
                }
            }
        }

        if (config_node["cam"]) {
            int cam_num = config_node["cam"].as<int>();
            for (int i = 0; i < cam_num; i++) {
                std::string id = "cam" + std::to_string(i);
                std::string id_name = id + "_calib";
                if (config_node[id_name]) {
                    auto cam_node =
                        YAML::LoadFile(config_node[id_name].as<std::string>());
                    if (cam_node) {
                        camera_calib[id] = camodocal::CameraFactory::instance()
                                               ->generateCameraFromYamlFile(
                                                   config_node[id_name]
                                                       .as<std::string>()
                                                       .c_str());

                        if (cam_node["camera_mask_path"] &&
                            cam_node["camera_mask_path"].as<std::string>() !=
                                "") {
                            std::string mask_path =
                                cam_node["camera_mask_path"].as<std::string>();
                            cv::Mat mask = cv::imread(mask_path.c_str(), -1);
                            camera_masks[id] = mask.clone();
                        } else {
                            camera_masks[id] = generateDefaultMask(id);
                        }
                    } else {
                        LOG(ERROR) << "no config_node[id_name]//cam";
                        std::exit(0);
                    }

                } else {
                    LOG(ERROR) << "no config_node[id_name]//cam";
                    std::exit(0);
                }

                std::string ex_name = "T_body_" + id;
                if (config_node[ex_name]) {
                    cv::Mat cv_T;
                    cv_params[ex_name.c_str()] >> cv_T;
                    Eigen::Matrix4d T_body_sensor;
                    cv::cv2eigen(cv_T, T_body_sensor);
                    addNewExternParam(id, T_body_sensor);
                } else {
                    LOG(ERROR) << "no config_node[ex_name]//cam";
                    std::exit(0);
                }

                use_cam_ids.push_back(id);

                { Tic[id] = extern_params["imu0"][id]; }
            }
        }

        if (config_node["camera_stereo"]) {
            for (auto &c : camera_calib) {
                std::string cam_id0 = c.first;
                if (config_node["camera_stereo"][cam_id0]) {
                    for (auto n : config_node["camera_stereo"][cam_id0]) {
                        std::string cam_id1 = n.as<std::string>();
                        camera_stereo.push_back(
                            std::pair<std::string, std::string>(cam_id0,
                                                                cam_id1));
                    }
                }
            }

            {
                std::cout << "use_cam_ids : " << std::endl;
                for (auto cam_id : use_cam_ids) {
                    std::cout << "  - " << cam_id << std::endl;
                }
                std::cout << "camera_stereo : " << std::endl;
                for (auto st : camera_stereo) {
                    std::cout << "  - " << st.first << " | " << st.second
                              << std::endl;
                }
            }
        }

        if (config_node["lidar"]) {
            std::cout << "LidarCalib : " << std::endl;
            int lidar_num = config_node["lidar"].as<int>();
            for (int i = 0; i < lidar_num; i++) {
                std::string id = "lidar" + std::to_string(i);
                std::string id_name = id + "_calib";
                if (config_node[id_name]) {
                    lidar_calib[id] =
                        LidarCalibration::Ptr(new LidarCalibration(
                            config_node[id_name].as<std::string>()));
                } else {
                    LOG(ERROR) << "no config_node[id_name]//lidar";
                    std::exit(0);
                }

                std::string ex_name = "T_body_" + id;
                if (config_node[ex_name]) {
                    cv::Mat cv_T;
                    cv_params[ex_name.c_str()] >> cv_T;
                    Eigen::Matrix4d T_body_sensor;
                    cv::cv2eigen(cv_T, T_body_sensor);
                    addNewExternParam(id, T_body_sensor);
                } else {
                    LOG(ERROR) << "no config_node[ex_name]//lidar";
                    std::exit(0);
                }

                std::cout << id_name << " : " << std::endl;
                lidar_calib[id]->print();
            }
        }

        {
            std::cout << "extern_params : " << std::endl;
            for (auto sensor0 : extern_params) {
                std::cout << "  - " << sensor0.first << " : [";
                for (auto sensor1 : sensor0.second) {
                    std::cout << sensor1.first << "  ";
                }
                std::cout << "]" << std::endl;
            }
        }
    }

    void addNewExternParam(const std::string &id, const Eigen::Matrix4d &ex) {
        if (extern_params.find(id) == extern_params.end()) {
            std::map<std::string, Eigen::Matrix4d> params;
            auto ex_inv = ex.inverse();
            for (auto p : extern_params["body"]) {
                params[p.first] = ex_inv * p.second;
                extern_params[p.first][id] =
                    extern_params["body"][p.first].inverse() * ex;
            }
            extern_params["body"][id] = ex;
            extern_params[id] = params;
            extern_params[id]["body"] = ex_inv;
        }
    }

    cv::Mat generateDefaultMask(const std::string &cam_id) {
        cv::Mat mask = cv::Mat(camera_calib[cam_id]->imageHeight(),
                               camera_calib[cam_id]->imageWidth(), CV_8UC1,
                               cv::Scalar(255));
        return mask.clone();
    }
};

}  // namespace SensorFusion