#include "ekf_utils.h"

namespace SensorFusion {

// void EKFPropagation(std::shared_ptr<State> state, const
// std::vector<std::shared_ptr<Type>> &order_NEW,
//                                  const std::vector<std::shared_ptr<Type>>
//                                  &order_OLD, const Eigen::MatrixXd &Phi,
//                                  const Eigen::MatrixXd &Q) {

//     assert(!order_NEW.empty());
//     assert(!order_OLD.empty());

//     int size_order_NEW = order_NEW.at(0)->size();
//     for (size_t i = 0; i < order_NEW.size() - 1; i++) {
//         assert(order_NEW.at(i)->id() + order_NEW.at(i)->size() ==
//         order_NEW.at(i + 1)->id())); size_order_NEW += order_NEW.at(i +
//         1)->size();
//     }

//     // Size of the old phi matrix
//     int size_order_OLD = order_OLD.at(0)->size();
//     for (size_t i = 1; i < order_OLD.size(); i++) {
//         size_order_OLD += order_OLD.at(i)->size();
//     }

//     // Assert that we have correct sizes
//     assert(size_order_NEW == Phi.rows());
//     assert(size_order_OLD == Phi.cols());
//     assert(size_order_NEW == Q.cols());
//     assert(size_order_NEW == Q.rows());

//     // Start !!!

//     // Get the location in small phi for each measuring variable
//     int current_it = 0;
//     std::vector<int> Phi_id;
//     for (const auto &var : order_OLD) {
//         Phi_id.push_back(current_it);
//         current_it += var->size();
//     }

//     // Loop through all our old states and get the state transition times it
//     // Cov_PhiT = [ Pxx ] [ Phi' ]'
//     Eigen::MatrixXd Cov_PhiT = Eigen::MatrixXd::Zero(state->_Cov.rows(),
//     Phi.rows()); for (size_t i = 0; i < order_OLD.size(); i++) {
//         std::shared_ptr<Type> var = order_OLD.at(i);
//         Cov_PhiT.noalias() +=
//             state->_Cov.block(0, var->id(), state->_Cov.rows(), var->size())
//             * Phi.block(0, Phi_id[i], Phi.rows(), var->size()).transpose();
//     }

//     // Get Phi_NEW*Covariance*Phi_NEW^t + Q
//     Eigen::MatrixXd Phi_Cov_PhiT = Q.selfadjointView<Eigen::Upper>();
//     for (size_t i = 0; i < order_OLD.size(); i++) {
//         std::shared_ptr<Type> var = order_OLD.at(i);
//         Phi_Cov_PhiT.noalias() += Phi.block(0, Phi_id[i], Phi.rows(),
//         var->size()) * Cov_PhiT.block(var->id(), 0, var->size(), Phi.rows());
//     }

//     // We are good to go!
//     int start_id = order_NEW.at(0)->id();
//     int phi_size = Phi.rows();
//     int total_size = state->_Cov.rows();
//     state->_Cov.block(start_id, 0, phi_size, total_size) =
//     Cov_PhiT.transpose(); state->_Cov.block(0, start_id, total_size,
//     phi_size) = Cov_PhiT; state->_Cov.block(start_id, start_id, phi_size,
//     phi_size) = Phi_Cov_PhiT;

//     // We should check if we are not positive semi-definitate (i.e. negative
//     diagionals is not s.p.d) Eigen::VectorXd diags = state->_Cov.diagonal();
//     bool found_neg = false;
//     for (int i = 0; i < diags.rows(); i++) {
//         if (diags(i) < 0.0) {
//             printf(RED "StateHelper::EKFPropagation() - diagonal at %d is
//             %.2f\n" RESET, i, diags(i)); found_neg = true;
//         }
//     }
//     assert(!found_neg);
// }

}  // namespace SensorFusion