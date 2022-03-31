#pragma once

#include "estimator/lidar/map_viewer.h"
#include "logger.h"
#include "visualizer/Visualizer.h"

namespace SensorFusion {

extern Visualizer::Visualizer::Ptr g_viz;
extern Logger::Ptr logger;
extern MapViewer map_viewer;
}  // namespace SensorFusion