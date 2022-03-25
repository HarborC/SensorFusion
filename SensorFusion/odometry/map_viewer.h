#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_set>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "utils/lidar_utils.h"

namespace SensorFusion {

class MapViewer {
public:
    using Ptr = std::shared_ptr<MapViewer>;
    MapViewer() { viewer_thread.reset(new std::thread(&MapViewer::run, this)); }
    ~MapViewer() {
        if (viewer_thread)
            viewer_thread->join();
    }

    void run() {
        viewer.reset(new pcl::visualization::PCLVisualizer("map_viewer"));
        viewer->setBackgroundColor(0.0, 0.0, 0.0);
        viewer->addCoordinateSystem(1.0);
        // viewer->setCameraPosition(0,0,200);
        viewer->initCameraParameters();

        lo_pose_nodes.reset(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            lo_pose_color(lo_pose_nodes, 255, 255, 255);
        viewer->addPointCloud(lo_pose_nodes, lo_pose_color, lo_pose_name);
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, lo_pose_size,
            lo_pose_name);

        while (!viewer->wasStopped()) {
            pointcloud_mutex.lock();
            showPointCloud(localmap_names, localmap_rgbs, localmap,
                           localmap_node_size);
            pointcloud_mutex.unlock();

            pointcloud_mutex.lock();
            showPointCloud(curpoint_names, curpoint_rgbs, curpoint,
                           curpoint_node_size);
            pointcloud_mutex.unlock();

            pointcloud_mutex.lock();
            showPoses(lo_pose_name, lo_pose_timestamps, lo_poses, lo_pose_nodes,
                      lo_pose_size);
            pointcloud_mutex.unlock();

            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::microseconds(10000));
        }
    }

    void addLocalMap(const pcl::PointCloud<PointType>::Ptr &cloud,
                     const std::string &name, int r, int g, int b) {
        pointcloud_mutex.lock();
        if (localmap_names.find(name) != localmap_names.end())
            localmap_names[name] = true;
        else {
            localmap[name] = pcl::PointCloud<PointType>::Ptr(
                new pcl::PointCloud<PointType>());
            localmap_rgbs[name] = std::vector<int>();
        }

        *(localmap[name]) = *cloud;
        localmap_rgbs[name] = {r, g, b};

        pointcloud_mutex.unlock();
    }

    void addCurrPoints(const pcl::PointCloud<PointType>::Ptr &cloud,
                       const std::string &name, int r, int g, int b) {
        pointcloud_mutex.lock();
        if (curpoint_names.find(name) != curpoint_names.end())
            curpoint_names[name] = true;
        else {
            curpoint[name] = pcl::PointCloud<PointType>::Ptr(
                new pcl::PointCloud<PointType>());
            curpoint_rgbs[name] = std::vector<int>();
        }

        *(curpoint[name]) = *cloud;
        curpoint_rgbs[name] = {r, g, b};

        pointcloud_mutex.unlock();
    }

    void addPose(const double &ts, const Eigen::Matrix4d &pose) {
        pointcloud_mutex.lock();
        if (lo_poses.find(ts) != lo_poses.end())
            lo_poses[ts] = pose;
        else {
            lo_poses[ts] = pose;
            lo_pose_timestamps.push_back(ts);
        }
        pointcloud_mutex.unlock();
    }

    void showPoses(std::string pose_name, std::vector<double> &pose_timestamps,
                   std::unordered_map<double, Eigen::Matrix4d> &poses,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr &pose_nodes,
                   int &pose_size) {
        viewer->removePointCloud(pose_name);
        pose_nodes.reset(new pcl::PointCloud<pcl::PointXYZ>);
        for (int i = 0; i < pose_timestamps.size(); i++) {
            auto &p = poses[pose_timestamps[i]];
            pcl::PointXYZ pt_temp;
            pt_temp.x = p(0, 3);
            pt_temp.y = p(1, 3);
            pt_temp.z = p(2, 3);
            pose_nodes->points.push_back(pt_temp);
        }

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            pose_color(pose_nodes, 255, 255, 255);
        viewer->addPointCloud(pose_nodes, pose_color, pose_name);
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pose_size,
            pose_name);
    }

    void showPointCloud(
        std::unordered_map<std::string, bool> &names,
        std::unordered_map<std::string, std::vector<int>> &colors,
        std::unordered_map<std::string, pcl::PointCloud<PointType>::Ptr>
            &pointclouds,
        int &node_size) {
        for (auto ppp : pointclouds) {
            const auto name = ppp.first;
            if (names.find(name) != names.end()) {
                if (names[name]) {
                    viewer->removePointCloud(name);
                } else {
                    continue;
                }
            }

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
                new pcl::PointCloud<pcl::PointXYZRGB>);
            const auto &pts = ppp.second->points;
            std::vector<int> &color = colors[name];
            for (int i = 0; i < pts.size(); i++) {
                const auto &point_in = pts[i];
                pcl::PointXYZRGB point;
                point.x = point_in.x;
                point.y = point_in.y;
                point.z = point_in.z;
                point.r = color[0];
                point.g = color[1];
                point.b = color[2];
                cloud->points.push_back(point);
            }

            viewer->addPointCloud<pcl::PointXYZRGB>(cloud, name);
            viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, node_size, name);
            names[name] = false;
        }
    }

    std::shared_ptr<std::thread> viewer_thread;
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    std::mutex pointcloud_mutex;

    // localmap
    std::unordered_map<std::string, bool> localmap_names;
    std::unordered_map<std::string, std::vector<int>> localmap_rgbs;
    std::unordered_map<std::string, pcl::PointCloud<PointType>::Ptr> localmap;
    int localmap_node_size = 2;

    // curpoint
    std::unordered_map<std::string, bool> curpoint_names;
    std::unordered_map<std::string, std::vector<int>> curpoint_rgbs;
    std::unordered_map<std::string, pcl::PointCloud<PointType>::Ptr> curpoint;
    int curpoint_node_size = 4;

    // Pose
    std::string lo_pose_name = "poses";
    std::vector<double> lo_pose_timestamps;
    std::unordered_map<double, Eigen::Matrix4d> lo_poses;
    pcl::PointCloud<pcl::PointXYZ>::Ptr lo_pose_nodes;
    int lo_pose_size = 8;
};

}  // namespace SensorFusion