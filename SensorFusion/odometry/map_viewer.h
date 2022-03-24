#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_set>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "utils/lidar_utils.h"

namespace SensorFusion {

class MapViewer {
public:
    using Ptr = std::shared_ptr<MapViewer>;
    MapViewer() {
        viewer_thread.reset(new std::thread(&MapViewer::run, this));
    }
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

        while (!viewer->wasStopped()) {
            
            pointcloud_mutex.lock();
            showPointCloud(localmap_names, localmap_rgbs, localmap, localmap_node_size);
            pointcloud_mutex.unlock();
            
            pointcloud_mutex.lock();
            showPointCloud(curpoint_names, curpoint_rgbs, curpoint, curpoint_node_size);
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
            localmap[name] = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
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
            curpoint[name] = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
            curpoint_rgbs[name] = std::vector<int>();
        }
        
        *(curpoint[name]) = *cloud;
        curpoint_rgbs[name] = {r, g, b};

        pointcloud_mutex.unlock();
    }

    void showPointCloud(std::unordered_map<std::string, bool> &names,
                        const std::unordered_map<std::string, std::vector<int>> &colors, 
                        const std::unordered_map<std::string, pcl::PointCloud<PointType>::Ptr>& pointclouds,
                        const int node_size) {
        for (auto ppp : pointclouds) {
            const auto name = ppp.first;
            if (names.find(name) != names.end()) {
                if (names[name]) {
                    viewer->removePointCloud(name);
                } else {
                    continue;
                }
            }

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            const auto& pts = ppp.second->points;
            for (int i = 0; i < pts.size(); i++) {
                const auto& point_in = pts[i];
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
    std::unordered_map<std::string, pcl::PointCloud<PointType>::Ptr>
        localmap;
    int localmap_node_size = 3;

    // curpoint
    std::unordered_map<std::string, bool> curpoint_names;
    std::unordered_map<std::string, std::vector<int>> curpoint_rgbs;
    std::unordered_map<std::string, pcl::PointCloud<PointType>::Ptr>
        curpoint;
    int curpoint_node_size = 5;
};

}  // namespace SensorFusion