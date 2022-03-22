/*
 * @Author: Jiagang Chen
 * @Date: 2021-12-09 17:41:59
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-12-09 17:47:03
 * @Description: ...
 * @Reference: ...
 */
#include "Visualizer.h"
#include "Pose.h"

void read_file(const std::string &path, std::vector<Visualizer::Pose> &poses) {

    poses.clear();

    std::ifstream f_read;
    f_read.open(path.c_str());

    while (!f_read.eof()) {
        std::string s;
        std::getline(f_read, s);
        if (s[0] == '#' || s == "")
            continue;

        if (!s.empty()) {
            std::string item;
            size_t trans = 0;
            double data_item[8];
            int count = 0;
            while ((trans = s.find(' ')) != std::string::npos) {
                item = s.substr(0, trans);
                data_item[count++] = std::stod(item);
                s.erase(0, trans + 1);
            }
            item = s.substr(0, trans);
            data_item[7] = std::stod(item);

            Visualizer::Pose pose;
            pose.timestamp_ = data_item[0] * 1e-9;
            pose.t_ = Eigen::Vector3d(data_item[1], data_item[2], data_item[3]);
            pose.R_ = Eigen::Quaterniond(data_item[7], data_item[4], data_item[5], data_item[6]).toRotationMatrix();
            poses.push_back(pose);
        }
    }

}

int main(int argc, const char** argv) {

    std::vector<Visualizer::Pose> imu_pose;
    read_file(std::string(argv[1]), imu_pose);
    int interval = 20;
    if (argc == 2)
        interval = std::stoi(std::string(argv[2]));

    Visualizer::Visualizer::Config viz_config;
    std::shared_ptr<Visualizer::Visualizer> viz = std::make_shared<Visualizer::Visualizer>(viz_config);

    // for (const auto& pose : imu_pose) {
    //     viz->DrawImuVisualizer::Pose(pose);
    // }
    std::vector<Visualizer::Pose> cam_pose;
    for (int i = 0; i < imu_pose.size(); i=i+interval)
        cam_pose.push_back(imu_pose[i]);
    viz->DrawCameras(cam_pose);

    return 0;
}