#pragma once

#include "logger.h"
#include "visualizer/Visualizer.h"

namespace SensorFusion {

extern Visualizer::Visualizer::Ptr g_viz;
extern Logger::Ptr logger;
}  // namespace SensorFusion