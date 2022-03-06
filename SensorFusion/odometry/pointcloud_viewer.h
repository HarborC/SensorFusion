#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_set>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace SensorFusion {

template <class POINT_TYPE>
class PointCloudViewer {
public:
    using Ptr = std::shared_ptr<PointCloudViewer>;
    PointCloudViewer() {
        viewer_thread.reset(new std::thread(&PointCloudViewer::run, this));
    }
    ~PointCloudViewer() {
        if (viewer_thread)
            viewer_thread->join();
    }

    void run() {
        viewer.reset(new pcl::visualization::PCLVisualizer("viewer"));
        viewer->setBackgroundColor(0.0, 0.0, 0.0);
        viewer->addCoordinateSystem(1.0);
        // viewer->setCameraPosition(0,0,200);
        viewer->initCameraParameters();

        while (!viewer->wasStopped()) {
            {
                pointcloud_mutex.lock();
                // Show PointCloud
                for (auto pt : pointcloud_map) {
                    const auto name = pt.first;
                    showPointCloud(pt.second, pt.first);
                }
                pointcloud_mutex.unlock();
            }

            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::microseconds(10000));
        }
    }

    void addPointCloud(const typename pcl::PointCloud<POINT_TYPE>::Ptr &cloud,
                       const std::string &name) {
        pointcloud_mutex.lock();
        if (pointcloud_names.find(name) != pointcloud_names.end())
            pointcloud_names[name] = true;
        pointcloud_map[name] = cloud;
        pointcloud_mutex.unlock();
    }

    void showPointCloud(const typename pcl::PointCloud<POINT_TYPE>::Ptr &cloud,
                        const std::string &name) {
        pcl::visualization::PointCloudColorHandlerRGBField<POINT_TYPE> points(
            cloud);
        if (pointcloud_names.find(name) != pointcloud_names.end()) {
            if (pointcloud_names[name]) {
                viewer->removePointCloud(name);
            } else {
                return;
            }
        }

        viewer->addPointCloud<POINT_TYPE>(cloud, points, name);
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, name);
        pointcloud_names[name] = false;
    }

    std::shared_ptr<std::thread> viewer_thread;
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    std::mutex pointcloud_mutex;
    std::unordered_map<std::string, bool> pointcloud_names;
    std::unordered_map<std::string, typename pcl::PointCloud<POINT_TYPE>::Ptr>
        pointcloud_map;
};

}  // namespace SensorFusion