#ifndef FAST_GICP_GICP_SETTINGS_HPP
#define FAST_GICP_GICP_SETTINGS_HPP

namespace fast_gicp {

enum class RegularizationMethod {
    NONE = 0,
    MIN_EIG = 1,
    NORMALIZED_MIN_EIG = 2,
    PLANE = 3,
    FROBENIUS = 4
};

enum class NeighborSearchMethod {
    DIRECT27 = 0,
    DIRECT7 = 1,
    DIRECT1 = 2,
    /* supported on only VGICP_CUDA */ DIRECT_RADIUS = 3
};

enum class VoxelAccumulationMode {
    ADDITIVE = 0,
    ADDITIVE_WEIGHTED = 1,
    MULTIPLICATIVE = 2
};
}  // namespace fast_gicp

#endif