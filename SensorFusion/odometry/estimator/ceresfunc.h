#pragma once

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <pthread.h>
#include <unordered_map>
#include <utility>
#include "../imu_integrator.h"
#include "../utils/math_utils.hpp"

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

/** \brief Ceres Cost Funtion between Lidar Pose and Ground
 */
struct Cost_NavState_PR_Ground {
    Cost_NavState_PR_Ground(Eigen::VectorXf ground_plane_coeff_)
        : ground_plane_coeff(ground_plane_coeff_) {
        // std::cout << ground_plane_coeff.transpose() << std::endl;
    }

    template <typename T>
    bool operator()(const T *pri_, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> PRi(pri_);
        Eigen::Matrix<T, 3, 1> Pi = PRi.template segment<3>(0);
        Sophus::SO3<T> SO3_Ri = Sophus::SO3<T>::exp(PRi.template segment<3>(3));
        Eigen::Map<Eigen::Matrix<T, 3, 1>> eResiduals(residual);
        eResiduals = Eigen::Matrix<T, 3, 1>::Zero();

        Eigen::Matrix<T, 4, 4> T_wl = Eigen::Matrix<T, 4, 4>::Identity();
        T_wl.topLeftCorner(3, 3) = SO3_Ri.matrix();
        T_wl.topRightCorner(3, 1) = Pi;
        Eigen::Matrix<T, 4, 4> T_lw = T_wl.inverse();
        Eigen::Matrix<T, 3, 3> R_lw = T_lw.topLeftCorner(3, 3);
        Eigen::Matrix<T, 3, 1> t_lw = T_lw.topRightCorner(3, 1);

        Eigen::Matrix<T, 4, 1> ground_plane_coeff_temp =
            ground_plane_coeff.cast<T>().template segment<4>(0);
        Eigen::Matrix<T, 4, 1> local_ground_plane;
        local_ground_plane.template segment<3>(0) =
            R_lw * init_ground_plane_coeff.cast<T>().template segment<3>(0);
        local_ground_plane.template segment<1>(3) =
            init_ground_plane_coeff.cast<T>().template segment<1>(3) -
            t_lw.transpose() * local_ground_plane.template segment<3>(0);
        eResiduals =
            SensorFusion::ominus(ground_plane_coeff_temp, local_ground_plane);
        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(Eigen::VectorXf ground_plane_coeff) {
        return (new ceres::AutoDiffCostFunction<Cost_NavState_PR_Ground, 3, 6>(
            new Cost_NavState_PR_Ground(ground_plane_coeff)));
    }
    Eigen::VectorXf ground_plane_coeff;
    static Eigen::Matrix<double, 3, 3> sqrt_information;
    static Eigen::VectorXf init_ground_plane_coeff;
};

/** \brief Ceres Cost Funtion between Lidar Pose and IMU Preintegration
 */
struct Cost_NavState_PRV_Bias {
    Cost_NavState_PRV_Bias(SensorFusion::IMUIntegrator &measure_,
                           Eigen::Vector3d &GravityVec_,
                           Eigen::Matrix<double, 15, 15> sqrt_information_)
        : imu_measure(measure_),
          GravityVec(GravityVec_),
          sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *pri_, const T *velobiasi_, const T *prj_,
                    const T *velobiasj_, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> PRi(pri_);
        Eigen::Matrix<T, 3, 1> Pi = PRi.template segment<3>(0);
        Sophus::SO3<T> SO3_Ri = Sophus::SO3<T>::exp(PRi.template segment<3>(3));

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> PRj(prj_);
        Eigen::Matrix<T, 3, 1> Pj = PRj.template segment<3>(0);
        Sophus::SO3<T> SO3_Rj = Sophus::SO3<T>::exp(PRj.template segment<3>(3));

        Eigen::Map<const Eigen::Matrix<T, 9, 1>> velobiasi(velobiasi_);
        Eigen::Matrix<T, 3, 1> Vi = velobiasi.template segment<3>(0);
        Eigen::Matrix<T, 3, 1> dbgi = velobiasi.template segment<3>(3) -
                                      imu_measure.GetBiasGyr().cast<T>();
        Eigen::Matrix<T, 3, 1> dbai = velobiasi.template segment<3>(6) -
                                      imu_measure.GetBiasAcc().cast<T>();

        Eigen::Map<const Eigen::Matrix<T, 9, 1>> velobiasj(velobiasj_);
        Eigen::Matrix<T, 3, 1> Vj = velobiasj.template segment<3>(0);

        Eigen::Map<Eigen::Matrix<T, 15, 1>> eResiduals(residual);
        eResiduals = Eigen::Matrix<T, 15, 1>::Zero();

        T dTij = T(imu_measure.GetDeltaTime());
        T dT2 = dTij * dTij;
        Eigen::Matrix<T, 3, 1> dPij = imu_measure.GetDeltaP().cast<T>();
        Eigen::Matrix<T, 3, 1> dVij = imu_measure.GetDeltaV().cast<T>();
        Sophus::SO3<T> dRij = Sophus::SO3<T>(imu_measure.GetDeltaQ().cast<T>());
        Sophus::SO3<T> RiT = SO3_Ri.inverse();

        Eigen::Matrix<T, 3, 1> rPij =
            RiT * (Pj - Pi - Vi * dTij - 0.5 * GravityVec.cast<T>() * dT2) -
            (dPij +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_P,
                                  SensorFusion::IMUIntegrator::O_BG)
                     .cast<T>() *
                 dbgi +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_P,
                                  SensorFusion::IMUIntegrator::O_BA)
                     .cast<T>() *
                 dbai);

        Sophus::SO3<T> dR_dbg = Sophus::SO3<T>::exp(
            imu_measure.GetJacobian()
                .block<3, 3>(SensorFusion::IMUIntegrator::O_R,
                             SensorFusion::IMUIntegrator::O_BG)
                .cast<T>() *
            dbgi);
        Sophus::SO3<T> rRij = (dRij * dR_dbg).inverse() * RiT * SO3_Rj;
        Eigen::Matrix<T, 3, 1> rPhiij = rRij.log();

        Eigen::Matrix<T, 3, 1> rVij =
            RiT * (Vj - Vi - GravityVec.cast<T>() * dTij) -
            (dVij +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_V,
                                  SensorFusion::IMUIntegrator::O_BG)
                     .cast<T>() *
                 dbgi +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_V,
                                  SensorFusion::IMUIntegrator::O_BA)
                     .cast<T>() *
                 dbai);

        eResiduals.template segment<3>(0) = rPij;
        eResiduals.template segment<3>(3) = rPhiij;
        eResiduals.template segment<3>(6) = rVij;
        eResiduals.template segment<6>(9) =
            velobiasj.template segment<6>(3) - velobiasi.template segment<6>(3);

        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(
        SensorFusion::IMUIntegrator &measure_, Eigen::Vector3d &GravityVec_,
        Eigen::Matrix<double, 15, 15> sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_NavState_PRV_Bias, 15, 6, 9, 6,
                                            9>(new Cost_NavState_PRV_Bias(
                measure_, GravityVec_, std::move(sqrt_information_))));
    }

    SensorFusion::IMUIntegrator imu_measure;
    Eigen::Vector3d GravityVec;
    Eigen::Matrix<double, 15, 15> sqrt_information;
};

/** \brief Ceres Cost Funtion between PointCloud Sharp Feature and Map Cloud
 */
struct Cost_NavState_IMU_Line {
    Cost_NavState_IMU_Line(Eigen::Vector3d _p, Eigen::Vector3d _vtx1,
                           Eigen::Vector3d _vtx2,
                           Eigen::Matrix<double, 1, 1> sqrt_information_)
        : point(std::move(_p)),
          vtx1(std::move(_vtx1)),
          vtx2(std::move(_vtx2)),
          sqrt_information(std::move(sqrt_information_)) {
        l12 = std::sqrt((vtx1(0) - vtx2(0)) * (vtx1(0) - vtx2(0)) +
                        (vtx1(1) - vtx2(1)) * (vtx1(1) - vtx2(1)) +
                        (vtx1(2) - vtx2(2)) * (vtx1(2) - vtx2(2)));
    }

    template <typename T>
    bool operator()(const T *PRi, const T *Exbl, T *residual) const {
        Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};
        Eigen::Matrix<T, 3, 1> lpa{T(vtx1.x()), T(vtx1.y()), T(vtx1.z())};
        Eigen::Matrix<T, 3, 1> lpb{T(vtx2.x()), T(vtx2.y()), T(vtx2.z())};

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
        Eigen::Quaternion<T> q_wb =
            Sophus::SO3<T>::exp(pri_wb.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wb = pri_wb.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_bl(Exbl);
        Eigen::Quaternion<T> qbl =
            Sophus::SO3<T>::exp(pri_bl.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> Pbl = pri_bl.template segment<3>(0);

        Eigen::Quaternion<T> q_wl = q_wb * qbl;
        Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl + t_wb;
        Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

        T a012 =
            ceres::sqrt(((P_to_Map(0) - lpa(0)) * (P_to_Map(1) - lpb(1)) -
                         (P_to_Map(0) - lpb(0)) * (P_to_Map(1) - lpa(1))) *
                            ((P_to_Map(0) - lpa(0)) * (P_to_Map(1) - lpb(1)) -
                             (P_to_Map(0) - lpb(0)) * (P_to_Map(1) - lpa(1))) +
                        ((P_to_Map(0) - lpa(0)) * (P_to_Map(2) - lpb(2)) -
                         (P_to_Map(0) - lpb(0)) * (P_to_Map(2) - lpa(2))) *
                            ((P_to_Map(0) - lpa(0)) * (P_to_Map(2) - lpb(2)) -
                             (P_to_Map(0) - lpb(0)) * (P_to_Map(2) - lpa(2))) +
                        ((P_to_Map(1) - lpa(1)) * (P_to_Map(2) - lpb(2)) -
                         (P_to_Map(1) - lpb(1)) * (P_to_Map(2) - lpa(2))) *
                            ((P_to_Map(1) - lpa(1)) * (P_to_Map(2) - lpb(2)) -
                             (P_to_Map(1) - lpb(1)) * (P_to_Map(2) - lpa(2))));
        T ld2 = a012 / T(l12);
        T _weight =
            T(1) - T(0.9) * ceres::abs(ld2) /
                       ceres::sqrt(ceres::sqrt(P_to_Map(0) * P_to_Map(0) +
                                               P_to_Map(1) * P_to_Map(1) +
                                               P_to_Map(2) * P_to_Map(2)));
        residual[0] = T(sqrt_information(0)) * _weight * ld2;
        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &curr_point_,
        const Eigen::Vector3d &last_point_a_,
        const Eigen::Vector3d &last_point_b_,
        Eigen::Matrix<double, 1, 1> sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_NavState_IMU_Line, 1, 6, 6>(
                new Cost_NavState_IMU_Line(curr_point_, last_point_a_,
                                           last_point_b_,
                                           std::move(sqrt_information_))));
    }

    Eigen::Vector3d point;
    Eigen::Vector3d vtx1;
    Eigen::Vector3d vtx2;
    double l12;
    Eigen::Matrix<double, 1, 1> sqrt_information;
};

/** \brief Ceres Cost Funtion between PointCloud Flat Feature and Map Cloud
 */
struct Cost_NavState_IMU_Plan {
    Cost_NavState_IMU_Plan(Eigen::Vector3d _p, double _pa, double _pb,
                           double _pc, double _pd,
                           Eigen::Matrix<double, 1, 1> sqrt_information_)
        : point(std::move(_p)),
          pa(_pa),
          pb(_pb),
          pc(_pc),
          pd(_pd),
          sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *PRi, const T *Exbl, T *residual) const {
        Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
        Eigen::Quaternion<T> q_wb =
            Sophus::SO3<T>::exp(pri_wb.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wb = pri_wb.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_bl(Exbl);
        Eigen::Quaternion<T> qbl =
            Sophus::SO3<T>::exp(pri_bl.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> Pbl = pri_bl.template segment<3>(0);

        Eigen::Quaternion<T> q_wl = q_wb * qbl;
        Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl + t_wb;
        Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

        T pd2 = T(pa) * P_to_Map(0) + T(pb) * P_to_Map(1) +
                T(pc) * P_to_Map(2) + T(pd);
        T _weight =
            T(1) - T(0.9) * ceres::abs(pd2) /
                       ceres::sqrt(ceres::sqrt(P_to_Map(0) * P_to_Map(0) +
                                               P_to_Map(1) * P_to_Map(1) +
                                               P_to_Map(2) * P_to_Map(2)));
        residual[0] = T(sqrt_information(0)) * _weight * pd2;
        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &curr_point_, const double &pa_,
        const double &pb_, const double &pc_, const double &pd_,
        Eigen::Matrix<double, 1, 1> sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_NavState_IMU_Plan, 1, 6, 6>(
                new Cost_NavState_IMU_Plan(curr_point_, pa_, pb_, pc_, pd_,
                                           std::move(sqrt_information_))));
    }

    double pa, pb, pc, pd;
    Eigen::Vector3d point;
    Eigen::Matrix<double, 1, 1> sqrt_information;
};

/** \brief Ceres Cost Funtion between PointCloud Flat Feature and Map Cloud
 */
struct Cost_NavState_IMU_Plan_Vec {
    Cost_NavState_IMU_Plan_Vec(Eigen::Vector3d _p, Eigen::Vector3d _p_proj,
                               Eigen::Matrix<double, 3, 3> _sqrt_information)
        : point(std::move(_p)),
          point_proj(std::move(_p_proj)),
          sqrt_information(std::move(_sqrt_information)) {}

    template <typename T>
    bool operator()(const T *PRi, const T *Exbl, T *residual) const {
        Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};
        Eigen::Matrix<T, 3, 1> cp_proj{T(point_proj.x()), T(point_proj.y()),
                                       T(point_proj.z())};

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
        Eigen::Quaternion<T> q_wb =
            Sophus::SO3<T>::exp(pri_wb.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wb = pri_wb.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_bl(Exbl);
        Eigen::Quaternion<T> qbl =
            Sophus::SO3<T>::exp(pri_bl.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> Pbl = pri_bl.template segment<3>(0);

        Eigen::Quaternion<T> q_wl = q_wb * qbl;
        Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl + t_wb;
        Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

        Eigen::Map<Eigen::Matrix<T, 3, 1>> eResiduals(residual);
        eResiduals = P_to_Map - cp_proj;
        T _weight =
            T(1) - T(0.9) * (P_to_Map - cp_proj).norm() /
                       ceres::sqrt(ceres::sqrt(P_to_Map(0) * P_to_Map(0) +
                                               P_to_Map(1) * P_to_Map(1) +
                                               P_to_Map(2) * P_to_Map(2)));
        eResiduals *= _weight;
        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &p_proj_,
        const Eigen::Matrix<double, 3, 3> sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_NavState_IMU_Plan_Vec, 3, 6,
                                            6>(new Cost_NavState_IMU_Plan_Vec(
                curr_point_, p_proj_, sqrt_information_)));
    }

    Eigen::Vector3d point;
    Eigen::Vector3d point_proj;
    Eigen::Matrix<double, 3, 3> sqrt_information;
};

/** \brief Ceres Cost Funtion between PointCloud NonFeature and Map Cloud
 */
struct Cost_NonFeature_ICP {
    Cost_NonFeature_ICP(Eigen::Vector3d _p, double _pa, double _pb, double _pc,
                        double _pd,
                        Eigen::Matrix<double, 1, 1> sqrt_information_)
        : point(std::move(_p)),
          pa(_pa),
          pb(_pb),
          pc(_pc),
          pd(_pd),
          sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *PRi, const T *Exbl, T *residual) const {
        Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
        Eigen::Quaternion<T> q_wb =
            Sophus::SO3<T>::exp(pri_wb.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wb = pri_wb.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_bl(Exbl);
        Eigen::Quaternion<T> qbl =
            Sophus::SO3<T>::exp(pri_bl.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> Pbl = pri_bl.template segment<3>(0);

        Eigen::Quaternion<T> q_wl = q_wb * qbl;
        Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl + t_wb;
        Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

        T pd2 = T(pa) * P_to_Map(0) + T(pb) * P_to_Map(1) +
                T(pc) * P_to_Map(2) + T(pd);
        T _weight =
            T(1) - T(0.9) * ceres::abs(pd2) /
                       ceres::sqrt(ceres::sqrt(P_to_Map(0) * P_to_Map(0) +
                                               P_to_Map(1) * P_to_Map(1) +
                                               P_to_Map(2) * P_to_Map(2)));
        residual[0] = T(sqrt_information(0)) * _weight * pd2;

        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &curr_point_, const double &pa_,
        const double &pb_, const double &pc_, const double &pd_,
        Eigen::Matrix<double, 1, 1> sqrt_information_) {
        return (new ceres::AutoDiffCostFunction<Cost_NonFeature_ICP, 1, 6, 6>(
            new Cost_NonFeature_ICP(curr_point_, pa_, pb_, pc_, pd_,
                                    std::move(sqrt_information_))));
    }

    double pa, pb, pc, pd;
    Eigen::Vector3d point;
    Eigen::Matrix<double, 1, 1> sqrt_information;
};

/** \brief Ceres Cost Funtion for Initial Gravity Direction
 */
struct Cost_Initial_G {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Cost_Initial_G(Eigen::Vector3d acc_) : acc(acc_) {}

    template <typename T>
    bool operator()(const T *q, T *residual) const {
        Eigen::Matrix<T, 3, 1> acc_T = acc.cast<T>();
        Eigen::Quaternion<T> q_wg{q[0], q[1], q[2], q[3]};
        Eigen::Matrix<T, 3, 1> g_I{T(0), T(0), T(-9.805)};
        Eigen::Matrix<T, 3, 1> resi = q_wg * g_I - acc_T;
        residual[0] = resi[0];
        residual[1] = resi[1];
        residual[2] = resi[2];

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d acc_) {
        return (new ceres::AutoDiffCostFunction<Cost_Initial_G, 3, 4>(
            new Cost_Initial_G(acc_)));
    }

    Eigen::Vector3d acc;
};

/** \brief Ceres Cost Funtion of IMU Factor in LIO Initialization
 */
struct Cost_Initialization_IMU {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Cost_Initialization_IMU(SensorFusion::IMUIntegrator &measure_,
                            Eigen::Vector3d ri_, Eigen::Vector3d rj_,
                            Eigen::Vector3d dp_,
                            Eigen::Matrix<double, 9, 9> sqrt_information_)
        : imu_measure(measure_),
          ri(ri_),
          rj(rj_),
          dp(dp_),
          sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *rwg_, const T *vi_, const T *vj_, const T *ba_,
                    const T *bg_, const T *s_, T *residual) const {
        Eigen::Matrix<T, 3, 1> G_I{T(0), T(0), T(-9.805)};

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba(ba_);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg(bg_);
        Eigen::Matrix<T, 3, 1> dbg = bg - imu_measure.GetBiasGyr().cast<T>();
        Eigen::Matrix<T, 3, 1> dba = ba - imu_measure.GetBiasAcc().cast<T>();

        Sophus::SO3<T> SO3_Ri = Sophus::SO3<T>::exp(ri.cast<T>());
        Sophus::SO3<T> SO3_Rj = Sophus::SO3<T>::exp(rj.cast<T>());

        Eigen::Matrix<T, 3, 1> dP = dp.cast<T>();

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> rwg(rwg_);
        Sophus::SO3<T> SO3_Rwg = Sophus::SO3<T>::exp(rwg);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> vi(vi_);
        Eigen::Matrix<T, 3, 1> Vi = vi;
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> vj(vj_);
        Eigen::Matrix<T, 3, 1> Vj = vj;

        T scale = s_[0];

        Eigen::Map<Eigen::Matrix<T, 9, 1>> eResiduals(residual);
        eResiduals = Eigen::Matrix<T, 9, 1>::Zero();

        T dTij = T(imu_measure.GetDeltaTime());
        T dT2 = dTij * dTij;
        Eigen::Matrix<T, 3, 1> dPij = imu_measure.GetDeltaP().cast<T>();
        Eigen::Matrix<T, 3, 1> dVij = imu_measure.GetDeltaV().cast<T>();
        Sophus::SO3<T> dRij = Sophus::SO3<T>(imu_measure.GetDeltaQ().cast<T>());
        Sophus::SO3<T> RiT = SO3_Ri.inverse();

        Eigen::Matrix<T, 3, 1> rPij =
            RiT * (scale * (dP - Vi * dTij) - SO3_Rwg * G_I * dT2 * T(0.5)) -
            (dPij +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_P,
                                  SensorFusion::IMUIntegrator::O_BG)
                     .cast<T>() *
                 dbg +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_P,
                                  SensorFusion::IMUIntegrator::O_BA)
                     .cast<T>() *
                 dba);

        Sophus::SO3<T> dR_dbg = Sophus::SO3<T>::exp(
            imu_measure.GetJacobian()
                .block<3, 3>(SensorFusion::IMUIntegrator::O_R,
                             SensorFusion::IMUIntegrator::O_BG)
                .cast<T>() *
            dbg);
        Sophus::SO3<T> rRij = (dRij * dR_dbg).inverse() * RiT * SO3_Rj;
        Eigen::Matrix<T, 3, 1> rPhiij = rRij.log();

        Eigen::Matrix<T, 3, 1> rVij =
            RiT * (scale * (Vj - Vi) - SO3_Rwg * G_I * dTij) -
            (dVij +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_V,
                                  SensorFusion::IMUIntegrator::O_BG)
                     .cast<T>() *
                 dbg +
             imu_measure.GetJacobian()
                     .block<3, 3>(SensorFusion::IMUIntegrator::O_V,
                                  SensorFusion::IMUIntegrator::O_BA)
                     .cast<T>() *
                 dba);

        eResiduals.template segment<3>(0) = rPij;
        eResiduals.template segment<3>(3) = rPhiij;
        eResiduals.template segment<3>(6) = rVij;

        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(
        SensorFusion::IMUIntegrator &measure_, Eigen::Vector3d ri_,
        Eigen::Vector3d rj_, Eigen::Vector3d dp_,
        Eigen::Matrix<double, 9, 9> sqrt_information_) {
        return (new ceres::AutoDiffCostFunction<Cost_Initialization_IMU, 9, 3,
                                                3, 3, 3, 3, 1>(
            new Cost_Initialization_IMU(measure_, ri_, rj_, dp_,
                                        std::move(sqrt_information_))));
    }

    SensorFusion::IMUIntegrator imu_measure;
    Eigen::Vector3d ri;
    Eigen::Vector3d rj;
    Eigen::Vector3d dp;
    Eigen::Matrix<double, 9, 9> sqrt_information;
};

/** \brief Ceres Cost Funtion of IMU Biases and Velocity Prior Factor in LIO
 * Initialization
 */
struct Cost_Initialization_Prior_bv {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Cost_Initialization_Prior_bv(Eigen::Vector3d prior_,
                                 Eigen::Matrix3d sqrt_information_)
        : prior(prior_), sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *bv_, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bv(bv_);
        Eigen::Matrix<T, 3, 1> Bv = bv;

        Eigen::Matrix<T, 3, 1> prior_T(prior.cast<T>());
        Eigen::Matrix<T, 3, 1> prior_Bv = prior_T;

        Eigen::Map<Eigen::Matrix<T, 3, 1>> eResiduals(residual);
        eResiduals = Eigen::Matrix<T, 3, 1>::Zero();

        eResiduals = Bv - prior_Bv;

        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d prior_,
                                       Eigen::Matrix3d sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_Initialization_Prior_bv, 3, 3>(
                new Cost_Initialization_Prior_bv(
                    prior_, std::move(sqrt_information_))));
    }

    Eigen::Vector3d prior;
    Eigen::Matrix3d sqrt_information;
};

/** \brief Ceres Cost Funtion of Rwg Prior Factor in LIO Initialization
 */
struct Cost_Initialization_Prior_R {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Cost_Initialization_Prior_R(Eigen::Vector3d prior_,
                                Eigen::Matrix3d sqrt_information_)
        : prior(prior_), sqrt_information(std::move(sqrt_information_)) {}

    template <typename T>
    bool operator()(const T *r_wg_, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> r_wg(r_wg_);
        Eigen::Matrix<T, 3, 1> R_wg = r_wg;
        Sophus::SO3<T> SO3_R_wg = Sophus::SO3<T>::exp(R_wg);

        Eigen::Matrix<T, 3, 1> prior_T(prior.cast<T>());
        Sophus::SO3<T> prior_R_wg = Sophus::SO3<T>::exp(prior_T);

        Sophus::SO3<T> d_R = SO3_R_wg.inverse() * prior_R_wg;
        Eigen::Matrix<T, 3, 1> d_Phi = d_R.log();

        Eigen::Map<Eigen::Matrix<T, 3, 1>> eResiduals(residual);
        eResiduals = Eigen::Matrix<T, 3, 1>::Zero();

        eResiduals = d_Phi;

        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d prior_,
                                       Eigen::Matrix3d sqrt_information_) {
        return (
            new ceres::AutoDiffCostFunction<Cost_Initialization_Prior_R, 3, 3>(
                new Cost_Initialization_Prior_R(prior_,
                                                std::move(sqrt_information_))));
    }

    Eigen::Vector3d prior;
    Eigen::Matrix3d sqrt_information;
};

struct PriorFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PriorFactor(const Eigen::Matrix4d &pose) {
        q_prior_ = Sophus::SO3<double>(pose.block<3, 3>(0, 0));
        t_prior_ = pose.block<3, 1>(0, 3);
    }

    template <typename T>
    bool operator()(const T *Ex_Para, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> Ex_para(Ex_Para);
        Sophus::SO3<T> q_para =
            Sophus::SO3<T>::exp(Ex_para.template segment<3>(3));
        Eigen::Matrix<T, 3, 1> t_para = Ex_para.template segment<3>(0);

        Sophus::SO3<T> q_prior = q_prior_.cast<T>();
        Eigen::Matrix<T, 3, 1> t_prior = t_prior_.cast<T>();

        Sophus::SO3<T> d_R = q_prior.inverse() * q_para;
        Eigen::Matrix<T, 3, 1> d_r = d_R.log();
        Eigen::Matrix<T, 3, 1> d_t = t_para - t_prior;

        T info_t = T(1000);
        T info_r = T(1);

        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residual);
        residuals[0] = info_t * d_t(0);
        residuals[1] = info_t * d_t(1);
        residuals[2] = info_t * d_t(2);
        residuals[3] = info_r * d_r(0);
        residuals[4] = info_r * d_r(1);
        residuals[5] = info_r * d_r(2);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Matrix4d &pose) {
        return (new ceres::AutoDiffCostFunction<PriorFactor, 6, 6>(
            new PriorFactor(pose)));
    }

    Sophus::SO3<double> q_prior_;
    Eigen::Matrix<double, 3, 1> t_prior_;
};

struct ProjectionTwoFrameOneCamFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionTwoFrameOneCamFactor(
        const Eigen::Vector3d &pts_i_, const Eigen::Vector3d &pts_j_,
        Eigen::Matrix<double, 2, 2> sqrt_information_)
        : pts_i(pts_i_),
          pts_j(pts_j_),
          sqrt_information(std::move(sqrt_information_)) {
        Eigen::Vector3d b1, b2;
        Eigen::Vector3d a = pts_j.normalized();
        Eigen::Vector3d tmp(0, 0, 1);
        if (a == tmp)
            tmp << 1, 0, 0;
        b1 = (tmp - a * (a.transpose() * tmp)).normalized();
        b2 = a.cross(b1);
        tangent_base.block<1, 3>(0, 0) = b1.transpose();
        tangent_base.block<1, 3>(1, 0) = b2.transpose();
    }

    template <typename T>
    bool operator()(const T *PRi, const T *PRj, const T *Exbc, const T *invDep,
                    T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wbi(PRi);
        Eigen::Quaternion<T> q_wbi =
            Sophus::SO3<T>::exp(pri_wbi.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wbi = pri_wbi.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wbj(PRj);
        Eigen::Quaternion<T> q_wbj =
            Sophus::SO3<T>::exp(pri_wbj.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wbj = pri_wbj.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> ex_ic(Exbc);
        Eigen::Quaternion<T> qic =
            Sophus::SO3<T>::exp(ex_ic.template segment<3>(3)).unit_quaternion();
        Eigen::Matrix<T, 3, 1> tic = ex_ic.template segment<3>(0);

        T inv_dep_i = invDep[0];

        Eigen::Matrix<T, 3, 1> pts_i_td, pts_j_td;
        pts_i_td = pts_i.cast<T>();
        pts_j_td = pts_j.cast<T>();

        Eigen::Matrix<T, 3, 1> pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Matrix<T, 3, 1> pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Matrix<T, 3, 1> pts_w = q_wbi * pts_imu_i + t_wbi;
        Eigen::Matrix<T, 3, 1> pts_imu_j = q_wbj.inverse() * (pts_w - t_wbj);
        Eigen::Matrix<T, 3, 1> pts_camera_j = qic.inverse() * (pts_imu_j - tic);

        Eigen::Map<Eigen::Matrix<T, 2, 1>> eResiduals(residual);
        eResiduals = tangent_base.cast<T>() *
                     (pts_camera_j.normalized() - pts_j_td.normalized());
        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &pts_i_, const Eigen::Vector3d &pts_j_,
        Eigen::Matrix<double, 2, 2> sqrt_information_) {
        return (new ceres::AutoDiffCostFunction<ProjectionTwoFrameOneCamFactor,
                                                2, 6, 6, 6, 1>(
            new ProjectionTwoFrameOneCamFactor(pts_i_, pts_j_,
                                               sqrt_information_)));
    }

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    Eigen::Matrix<double, 2, 2> sqrt_information;
};

struct ProjectionTwoFrameTwoCamFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionTwoFrameTwoCamFactor(
        const Eigen::Vector3d &pts_i_, const Eigen::Vector3d &pts_j_,
        Eigen::Matrix<double, 2, 2> sqrt_information_)
        : pts_i(pts_i_),
          pts_j(pts_j_),
          sqrt_information(std::move(sqrt_information_)) {
        Eigen::Vector3d b1, b2;
        Eigen::Vector3d a = pts_j.normalized();
        Eigen::Vector3d tmp(0, 0, 1);
        if (a == tmp)
            tmp << 1, 0, 0;
        b1 = (tmp - a * (a.transpose() * tmp)).normalized();
        b2 = a.cross(b1);
        tangent_base.block<1, 3>(0, 0) = b1.transpose();
        tangent_base.block<1, 3>(1, 0) = b2.transpose();
    }

    template <typename T>
    bool operator()(const T *PRi, const T *PRj, const T *Exbci, const T *Exbcj,
                    const T *invDep, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wbi(PRi);
        Eigen::Quaternion<T> q_wbi =
            Sophus::SO3<T>::exp(pri_wbi.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wbi = pri_wbi.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wbj(PRj);
        Eigen::Quaternion<T> q_wbj =
            Sophus::SO3<T>::exp(pri_wbj.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> t_wbj = pri_wbj.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> ex_ici(Exbci);
        Eigen::Quaternion<T> qic1 =
            Sophus::SO3<T>::exp(ex_ici.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> tic1 = ex_ici.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> ex_icj(Exbcj);
        Eigen::Quaternion<T> qic2 =
            Sophus::SO3<T>::exp(ex_icj.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> tic2 = ex_icj.template segment<3>(0);

        T inv_dep_i = invDep[0];

        Eigen::Matrix<T, 3, 1> pts_i_td, pts_j_td;
        pts_i_td = pts_i.cast<T>();
        pts_j_td = pts_j.cast<T>();

        Eigen::Matrix<T, 3, 1> pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Matrix<T, 3, 1> pts_imu_i = qic1 * pts_camera_i + tic1;
        Eigen::Matrix<T, 3, 1> pts_w = q_wbi * pts_imu_i + t_wbi;
        Eigen::Matrix<T, 3, 1> pts_imu_j = q_wbj.inverse() * (pts_w - t_wbj);
        Eigen::Matrix<T, 3, 1> pts_camera_j =
            qic2.inverse() * (pts_imu_j - tic2);

        Eigen::Map<Eigen::Matrix<T, 2, 1>> eResiduals(residual);
        eResiduals = tangent_base.cast<T>() *
                     (pts_camera_j.normalized() - pts_j_td.normalized());
        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &pts_i_, const Eigen::Vector3d &pts_j_,
        Eigen::Matrix<double, 2, 2> sqrt_information_) {
        return (new ceres::AutoDiffCostFunction<ProjectionTwoFrameTwoCamFactor,
                                                2, 6, 6, 6, 6, 1>(
            new ProjectionTwoFrameTwoCamFactor(pts_i_, pts_j_,
                                               sqrt_information_)));
    }

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    Eigen::Matrix<double, 2, 2> sqrt_information;
};

struct ProjectionOneFrameTwoCamFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionOneFrameTwoCamFactor(
        const Eigen::Vector3d &pts_i_, const Eigen::Vector3d &pts_j_,
        Eigen::Matrix<double, 2, 2> sqrt_information_)
        : pts_i(pts_i_),
          pts_j(pts_j_),
          sqrt_information(std::move(sqrt_information_)) {
        Eigen::Vector3d b1, b2;
        Eigen::Vector3d a = pts_j.normalized();
        Eigen::Vector3d tmp(0, 0, 1);
        if (a == tmp)
            tmp << 1, 0, 0;
        b1 = (tmp - a * (a.transpose() * tmp)).normalized();
        b2 = a.cross(b1);
        tangent_base.block<1, 3>(0, 0) = b1.transpose();
        tangent_base.block<1, 3>(1, 0) = b2.transpose();
    }

    template <typename T>
    bool operator()(const T *Exbci, const T *Exbcj, const T *invDep,
                    T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> ex_ici(Exbci);
        Eigen::Quaternion<T> qic1 =
            Sophus::SO3<T>::exp(ex_ici.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> tic1 = ex_ici.template segment<3>(0);

        Eigen::Map<const Eigen::Matrix<T, 6, 1>> ex_icj(Exbcj);
        Eigen::Quaternion<T> qic2 =
            Sophus::SO3<T>::exp(ex_icj.template segment<3>(3))
                .unit_quaternion();
        Eigen::Matrix<T, 3, 1> tic2 = ex_icj.template segment<3>(0);

        T inv_dep_i = invDep[0];

        Eigen::Matrix<T, 3, 1> pts_i_td, pts_j_td;
        pts_i_td = pts_i.cast<T>();
        pts_j_td = pts_j.cast<T>();

        Eigen::Matrix<T, 3, 1> pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Matrix<T, 3, 1> pts_imu = qic1 * pts_camera_i + tic1;
        Eigen::Matrix<T, 3, 1> pts_camera_j = qic2.inverse() * (pts_imu - tic2);

        Eigen::Map<Eigen::Matrix<T, 2, 1>> eResiduals(residual);
        eResiduals = tangent_base.cast<T>() *
                     (pts_camera_j.normalized() - pts_j_td.normalized());
        eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d &pts_i_, const Eigen::Vector3d &pts_j_,
        Eigen::Matrix<double, 2, 2> sqrt_information_) {
        return (new ceres::AutoDiffCostFunction<ProjectionOneFrameTwoCamFactor,
                                                2, 6, 6, 1>(
            new ProjectionOneFrameTwoCamFactor(pts_i_, pts_j_,
                                               sqrt_information_)));
    }

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    Eigen::Matrix<double, 2, 2> sqrt_information;
};