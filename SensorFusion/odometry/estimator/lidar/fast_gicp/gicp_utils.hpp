#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>
#include <vector>

#include "../../../nano_gicp/nanoflann.hpp"
#include "gicp_voxel.hpp"

namespace fast_gicp {

template <typename PointT>
bool calculate_covariances(
    const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
    nanoflann::KdTreeFLANN<PointT>& kdtree,
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>&
        covariances,
    const RegularizationMethod regularization_method =
        RegularizationMethod::PLANE) {
    if (kdtree.getInputCloud() != cloud) {
        kdtree.setInputCloud(cloud);
    }
    covariances.resize(cloud->size());

#ifdef _OPENMP
    int max_num_threads = omp_get_max_threads();
#else
    int max_num_threads = 1;
#endif

    int k_correspondences = 20;

#pragma omp parallel for num_threads(max_num_threads) schedule(guided, 8)
    for (int i = 0; i < cloud->size(); i++) {
        std::vector<int> k_indices;
        std::vector<float> k_sq_distances;
        kdtree.nearestKSearch(cloud->at(i), k_correspondences, k_indices,
                              k_sq_distances);

        Eigen::Matrix<double, 4, -1> neighbors(4, k_correspondences);
        for (int j = 0; j < k_indices.size(); j++) {
            neighbors.col(j) = cloud->at(k_indices[j])
                                   .getVector4fMap()
                                   .template cast<double>();
        }

        neighbors.colwise() -= neighbors.rowwise().mean().eval();
        Eigen::Matrix4d cov =
            neighbors * neighbors.transpose() / k_correspondences;

        if (regularization_method == RegularizationMethod::NONE) {
            covariances[i] = cov;
        } else if (regularization_method == RegularizationMethod::FROBENIUS) {
            double lambda = 1e-3;
            Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() +
                                lambda * Eigen::Matrix3d::Identity();
            Eigen::Matrix3d C_inv = C.inverse();
            covariances[i].setZero();
            covariances[i].template block<3, 3>(0, 0) =
                (C_inv / C_inv.norm()).inverse();
        } else {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                cov.block<3, 3>(0, 0),
                Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Vector3d values;

            switch (regularization_method) {
            default:
                std::cerr << "here must not be reached" << std::endl;
                abort();
            case RegularizationMethod::PLANE:
                values = Eigen::Vector3d(1, 1, 1e-3);
                break;
            case RegularizationMethod::MIN_EIG:
                values = svd.singularValues().array().max(1e-3);
                break;
            case RegularizationMethod::NORMALIZED_MIN_EIG:
                values = svd.singularValues() / svd.singularValues().maxCoeff();
                values = values.array().max(1e-3);
                break;
            }

            covariances[i].setZero();
            covariances[i].template block<3, 3>(0, 0) =
                svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
        }
    }

    return true;
}

template <typename PointT>
void update_correspondences(
    const Eigen::Matrix4d& transform,
    const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
    const typename GaussianVoxelMap<PointT>::Ptr voxelmap,
    std::vector<std::pair<int, GaussianVoxel::Ptr>>& voxel_correspondences,
    NeighborSearchMethod search_method = NeighborSearchMethod::DIRECT7) {
    voxel_correspondences.clear();
    auto offsets = neighbor_offsets(search_method);

#ifdef _OPENMP
    int max_num_threads = omp_get_max_threads();
#else
    int max_num_threads = 1;
#endif

    std::vector<std::vector<std::pair<int, GaussianVoxel::Ptr>>> corrs(
        max_num_threads);
    for (auto& c : corrs) {
        c.reserve((cloud->size() * offsets.size()) / max_num_threads);
    }

#pragma omp parallel for num_threads(max_num_threads) schedule(guided, 8)
    for (int i = 0; i < cloud->size(); i++) {
        const Eigen::Vector4d mean_A =
            cloud->at(i).getVector4fMap().template cast<double>();
        Eigen::Vector4d transed_mean_A = transform * mean_A;
        Eigen::Vector3i coord = voxelmap->voxel_coord(transed_mean_A);

        for (const auto& offset : offsets) {
            auto voxel = voxelmap->lookup_voxel(coord + offset);
            if (voxel != nullptr) {
                corrs[omp_get_thread_num()].push_back(std::make_pair(i, voxel));
            }
        }
    }

    voxel_correspondences.reserve(cloud->size() * offsets.size());
    for (const auto& c : corrs) {
        voxel_correspondences.insert(voxel_correspondences.end(), c.begin(),
                                     c.end());
    }

    //     // precompute combined covariances
    //     voxel_mahalanobis_.resize(voxel_correspondences.size());

    // #pragma omp parallel for num_threads(max_num_threads) schedule(guided, 8)
    //     for (int i = 0; i < voxel_correspondences.size(); i++) {
    //         const auto& corr = voxel_correspondences[i];
    //         const auto& cov_A = source_covs_[corr.first];
    //         const auto& cov_B = corr.second->cov;

    //         Eigen::Matrix4d RCR = cov_B + transform * cov_A *
    //         transform.transpose(); RCR(3, 3) = 1.0;

    //         voxel_mahalanobis_[i] = RCR.inverse();
    //         voxel_mahalanobis_[i](3, 3) = 0.0;
    //     }
}

}  // namespace fast_gicp