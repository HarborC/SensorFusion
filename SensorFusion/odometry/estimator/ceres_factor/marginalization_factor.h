#pragma once

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <pthread.h>
#include <unordered_map>
#include <utility>

#include "sophus/so3.hpp"

const int NUM_THREADS = 4;

/** \brief Residual Block Used for marginalization
 */
struct ResidualBlockInfo {
    ResidualBlockInfo(ceres::CostFunction *_cost_function,
                      ceres::LossFunction *_loss_function,
                      std::vector<double *> _parameter_blocks,
                      std::vector<int> _drop_set)
        : cost_function(_cost_function),
          loss_function(_loss_function),
          parameter_blocks(std::move(_parameter_blocks)),
          drop_set(std::move(_drop_set)) {}

    void Evaluate() {
        residuals.resize(cost_function->num_residuals());

        std::vector<int> block_sizes = cost_function->parameter_block_sizes();
        raw_jacobians = new double *[block_sizes.size()];
        jacobians.resize(block_sizes.size());

        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
            jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
            raw_jacobians[i] = jacobians[i].data();
        }
        cost_function->Evaluate(parameter_blocks.data(), residuals.data(),
                                raw_jacobians);

        if (loss_function) {
            double residual_scaling_, alpha_sq_norm_;

            double sq_norm, rho[3];

            sq_norm = residuals.squaredNorm();
            loss_function->Evaluate(sq_norm, rho);

            double sqrt_rho1_ = sqrt(rho[1]);

            if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
                residual_scaling_ = sqrt_rho1_;
                alpha_sq_norm_ = 0.0;
            } else {
                const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
                const double alpha = 1.0 - sqrt(D);
                residual_scaling_ = sqrt_rho1_ / (1 - alpha);
                alpha_sq_norm_ = alpha / sq_norm;
            }

            for (int i = 0; i < static_cast<int>(parameter_blocks.size());
                 i++) {
                jacobians[i] =
                    sqrt_rho1_ *
                    (jacobians[i] - alpha_sq_norm_ * residuals *
                                        (residuals.transpose() * jacobians[i]));
            }

            residuals *= residual_scaling_;
        }
    }

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians{};
    std::vector<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        jacobians;
    Eigen::VectorXd residuals;
};

struct ThreadsStruct {
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size;
    std::unordered_map<long, int> parameter_block_idx;
};

/** \brief Multi-thread to process marginalization
 */
void *ThreadsConstructA(void *threadsstruct);

/** \brief marginalization infomation
 */
class MarginalizationInfo {
public:
    ~MarginalizationInfo() {
        //			ROS_WARN("release marginlizationinfo");

        for (auto it = parameter_block_data.begin();
             it != parameter_block_data.end(); ++it)
            delete[] it->second;

        for (int i = 0; i < (int)factors.size(); i++) {
            delete[] factors[i]->raw_jacobians;
            delete factors[i]->cost_function;
            delete factors[i];
        }
    }

    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info) {
        factors.emplace_back(residual_block_info);

        std::vector<double *> &parameter_blocks =
            residual_block_info->parameter_blocks;
        std::vector<int> parameter_block_sizes =
            residual_block_info->cost_function->parameter_block_sizes();

        for (int i = 0;
             i < static_cast<int>(residual_block_info->parameter_blocks.size());
             i++) {
            double *addr = parameter_blocks[i];
            int size = parameter_block_sizes[i];
            parameter_block_size[reinterpret_cast<long>(addr)] = size;
        }

        for (int i = 0;
             i < static_cast<int>(residual_block_info->drop_set.size()); i++) {
            double *addr = parameter_blocks[residual_block_info->drop_set[i]];
            parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
        }
    }

    void preMarginalize() {
        for (auto it : factors) {
            it->Evaluate();

            std::vector<int> block_sizes =
                it->cost_function->parameter_block_sizes();
            for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
                long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
                int size = block_sizes[i];
                if (parameter_block_data.find(addr) ==
                    parameter_block_data.end()) {
                    double *data = new double[size];
                    memcpy(data, it->parameter_blocks[i],
                           sizeof(double) * size);
                    parameter_block_data[addr] = data;
                }
            }
        }
    }

    void marginalize() {
        int pos = 0;
        for (auto &it : parameter_block_idx) {
            it.second = pos;
            pos += parameter_block_size[it.first];
        }

        m = pos;

        for (const auto &it : parameter_block_size) {
            if (parameter_block_idx.find(it.first) ==
                parameter_block_idx.end()) {
                parameter_block_idx[it.first] = pos;
                pos += it.second;
            }
        }

        n = pos - m;

        Eigen::MatrixXd A(pos, pos);
        Eigen::VectorXd b(pos);
        A.setZero();
        b.setZero();

        pthread_t tids[NUM_THREADS];
        ThreadsStruct threadsstruct[NUM_THREADS];
        int i = 0;
        for (auto it : factors) {
            threadsstruct[i].sub_factors.push_back(it);
            i++;
            i = i % NUM_THREADS;
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            threadsstruct[i].A = Eigen::MatrixXd::Zero(pos, pos);
            threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
            threadsstruct[i].parameter_block_size = parameter_block_size;
            threadsstruct[i].parameter_block_idx = parameter_block_idx;
            int ret = pthread_create(&tids[i], NULL, ThreadsConstructA,
                                     (void *)&(threadsstruct[i]));
            if (ret != 0) {
                std::cout << "pthread_create error" << std::endl;
                exit(1);
            }
        }
        for (int i = NUM_THREADS - 1; i >= 0; i--) {
            pthread_join(tids[i], NULL);
            A += threadsstruct[i].A;
            b += threadsstruct[i].b;
        }
        Eigen::MatrixXd Amm =
            0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

        Eigen::MatrixXd Amm_inv =
            saes.eigenvectors() *
            Eigen::VectorXd(
                (saes.eigenvalues().array() > eps)
                    .select(saes.eigenvalues().array().inverse(), 0))
                .asDiagonal() *
            saes.eigenvectors().transpose();

        Eigen::VectorXd bmm = b.segment(0, m);
        Eigen::MatrixXd Amr = A.block(0, m, m, n);
        Eigen::MatrixXd Arm = A.block(m, 0, n, m);
        Eigen::MatrixXd Arr = A.block(m, m, n, n);
        Eigen::VectorXd brr = b.segment(m, n);
        A = Arr - Arm * Amm_inv * Amr;
        b = brr - Arm * Amm_inv * bmm;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
        Eigen::VectorXd S =
            Eigen::VectorXd((saes2.eigenvalues().array() > eps)
                                .select(saes2.eigenvalues().array(), 0));
        Eigen::VectorXd S_inv = Eigen::VectorXd(
            (saes2.eigenvalues().array() > eps)
                .select(saes2.eigenvalues().array().inverse(), 0));

        Eigen::VectorXd S_sqrt = S.cwiseSqrt();
        Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

        linearized_jacobians =
            S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
        linearized_residuals =
            S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    }

    std::vector<double *> getParameterBlocks(
        std::unordered_map<long, double *> &addr_shift) {
        std::vector<double *> keep_block_addr;
        keep_block_size.clear();
        keep_block_idx.clear();
        keep_block_data.clear();

        for (const auto &it : parameter_block_idx) {
            if (it.second >= m) {
                keep_block_size.push_back(parameter_block_size[it.first]);
                keep_block_idx.push_back(parameter_block_idx[it.first]);
                keep_block_data.push_back(parameter_block_data[it.first]);
                keep_block_addr.push_back(addr_shift[it.first]);
            }
        }
        sum_block_size = std::accumulate(std::begin(keep_block_size),
                                         std::end(keep_block_size), 0);

        return keep_block_addr;
    }

    std::vector<ResidualBlockInfo *> factors;
    int m, n;
    std::unordered_map<long, int> parameter_block_size;
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx;
    std::unordered_map<long, double *> parameter_block_data;

    std::vector<int> keep_block_size;
    std::vector<int> keep_block_idx;
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
};

/** \brief Ceres Cost Funtion Used for Marginalization
 */
class MarginalizationFactor : public ceres::CostFunction {
public:
    explicit MarginalizationFactor(MarginalizationInfo *_marginalization_info)
        : marginalization_info(_marginalization_info) {
        int cnt = 0;
        for (auto it : marginalization_info->keep_block_size) {
            mutable_parameter_block_sizes()->push_back(it);
            cnt += it;
        }
        set_num_residuals(marginalization_info->n);
    };

    bool Evaluate(double const *const *parameters, double *residuals,
                  double **jacobians) const override {
        int n = marginalization_info->n;
        int m = marginalization_info->m;
        Eigen::VectorXd dx(n);
        for (int i = 0;
             i < static_cast<int>(marginalization_info->keep_block_size.size());
             i++) {
            int size = marginalization_info->keep_block_size[i];
            int idx = marginalization_info->keep_block_idx[i] - m;
            Eigen::VectorXd x =
                Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
            Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(
                marginalization_info->keep_block_data[i], size);
            if (size == 6) {
                dx.segment<3>(idx + 0) = x.segment<3>(0) - x0.segment<3>(0);
                dx.segment<3>(idx + 3) =
                    (Sophus::SO3d::exp(x.segment<3>(3)).inverse() *
                     Sophus::SO3d::exp(x0.segment<3>(3)))
                        .log();
            } else {
                dx.segment(idx, size) = x - x0;
            }
        }
        Eigen::Map<Eigen::VectorXd>(residuals, n) =
            marginalization_info->linearized_residuals +
            marginalization_info->linearized_jacobians * dx;
        if (jacobians) {
            for (int i = 0;
                 i <
                 static_cast<int>(marginalization_info->keep_block_size.size());
                 i++) {
                if (jacobians[i]) {
                    int size = marginalization_info->keep_block_size[i];
                    int idx = marginalization_info->keep_block_idx[i] - m;
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>
                        jacobian(jacobians[i], n, size);
                    jacobian.setZero();
                    jacobian.leftCols(size) =
                        marginalization_info->linearized_jacobians.middleCols(
                            idx, size);
                }
            }
        }
        return true;
    }

    MarginalizationInfo *marginalization_info;
};
