#include "global.h"

namespace SensorFusion {

Visualizer::Visualizer::Ptr g_viz = Visualizer::Visualizer::Ptr(
    new Visualizer::Visualizer(Visualizer::Visualizer::Config(), false));

Logger::Ptr logger = Logger::Ptr(new Logger());

}  // namespace SensorFusion