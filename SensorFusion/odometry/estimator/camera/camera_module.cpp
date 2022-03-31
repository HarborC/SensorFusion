#include "camera_module.h"
#include "../../global.h"
#include "../ceres_factor/projection_OneFrameTwoCam_factor.h"
#include "../ceres_factor/projection_TwoFrameOneCam_Factor.h"
#include "../ceres_factor/projection_TwoFrameTwoCam_Factor.h"
#include "initial_sfm.h"

#include <unistd.h>
#include <algorithm>

namespace SensorFusion {

double CameraModule::dataSynchronize(const double& timestamp) {
    if (timestamp > 0) {
        if (!curr_data) {
            if (data_queue->empty()) {
                sleep(1 * 1e-3);
                return 0;
            } else {
                data_queue->pop(curr_data);
            }
        }

        while (curr_data->timestamp <= timestamp) {
            CameraFrame::Ptr new_frame(new CameraFrame);
            new_frame->trackResult = curr_data;
            new_frame->timeStamp = curr_data->timestamp;
            curr_timestamp = curr_data->timestamp;

            f_manager.addFeature(frame_count++, curr_data->observations, 0.0);

            // Debug for Show
            {
                const auto track_result = f_manager.showFrame(frame_count - 1);
                for (const auto& im : track_result) {
                    const std::string& cam_id = im.first;
                    if (curr_data->input_images->img_data.find(cam_id) !=
                        curr_data->input_images->img_data.end())
                        cv::imshow(
                            cam_id + "_track_result",
                            utility::drawKeypoint1(
                                curr_data->input_images->img_data[cam_id],
                                im.second));
                }
                cv::waitKey(10);
            }

            std::vector<ImuData::Ptr> imu_meas;
            getPreIntegratedImuData(imu_meas);
            if (imu_meas.size()) {
                new_frame->pre_imu_enabled = true;
                new_frame->imuIntegrator.PushIMUMsg(imu_meas);
                new_frame->imuIntegrator.PreIntegration(
                    last_timestamp, curr_timestamp, prev_frame->ba,
                    prev_frame->bg);
            } else {
                new_frame->pre_imu_enabled = false;
            }

            window_frames.push_back(new_frame);

            last_timestamp = curr_timestamp;

            if (!data_queue->empty()) {
                data_queue->pop(curr_data);
            } else {
                curr_data = nullptr;
                break;
            }
        }
    } else {
        if (data_queue->empty()) {
            sleep(1 * 1e-3);
            return 0;
        } else {
            data_queue->pop(curr_data);
        }

        CameraFrame::Ptr new_frame(new CameraFrame);
        new_frame->trackResult = curr_data;
        new_frame->timeStamp = curr_data->timestamp;
        curr_timestamp = curr_data->timestamp;

        f_manager.addFeature(frame_count++, curr_data->observations, 0.0);

        {
            const auto track_result = f_manager.showFrame(frame_count - 1);
            for (const auto& im : track_result) {
                const std::string& cam_id = im.first;
                if (curr_data->input_images->img_data.find(cam_id) !=
                    curr_data->input_images->img_data.end())
                    cv::imshow(cam_id + "_track_result",
                               utility::drawKeypoint1(
                                   curr_data->input_images->img_data[cam_id],
                                   im.second));
            }
            cv::waitKey(10);
        }

        std::vector<ImuData::Ptr> imu_meas;
        getPreIntegratedImuData(imu_meas);
        if (imu_meas.size()) {
            new_frame->pre_imu_enabled = true;
            new_frame->imuIntegrator.PushIMUMsg(imu_meas);
            new_frame->imuIntegrator.PreIntegration(
                last_timestamp, curr_timestamp, prev_frame->ba, prev_frame->bg);
        } else {
            new_frame->pre_imu_enabled = false;
        }

        window_frames.push_back(new_frame);

        last_timestamp = curr_timestamp;
    }

    return last_timestamp;
}

int CameraModule::initialize() {
    return 0;

    // check imu observibility
    {
        if (window_frames.size() >= 2) {
            Eigen::Vector3d sum_g = Eigen::Vector3d::Zero();
            auto it = window_frames.begin();
            for (it = window_frames.begin(), it++; it != window_frames.end();
                 ++it) {
                sum_g += (*it)->imuIntegrator.GetDeltaV() /
                         (*it)->imuIntegrator.GetDeltaTime();
            }

            auto aver_g = sum_g / ((int)window_frames.size() - 1);

            double var = 0;
            for (it = window_frames.begin(), it++; it != window_frames.end();
                 ++it) {
                auto tmp_g = (*it)->imuIntegrator.GetDeltaV() /
                             (*it)->imuIntegrator.GetDeltaTime();
                var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            }
            var = std::sqrt(var / ((int)window_frames.size() - 1));

            if (var < 0.25) {
                std::cout << "imu excitation not enouth!" << std::endl;
                // return false;
            }
        } else {
            return 0;
        }
    }

    // global sfm
    {
        std::unordered_map<std::string, std::vector<SFMFeature>> sfm_feat;

        for (auto& it_per_id : f_manager.feature) {
            int imu_j = it_per_id.start_frame - 1;
            SFMFeature tmp_feature;
            tmp_feature.state = false;
            tmp_feature.id = it_per_id.feature_id;
            auto cam_id = it_per_id.feature_per_frame[0].camids[0];
            for (auto& it_per_frame : it_per_id.feature_per_frame) {
                imu_j++;
                Eigen::Vector3d pts_j = it_per_frame.points[0];
                tmp_feature.observation.push_back(std::make_pair(
                    imu_j, Eigen::Vector2d{pts_j.x() / pts_j.z(),
                                           pts_j.y() / pts_j.z()}));
            }
            if (sfm_feat.find(cam_id) == sfm_feat.end())
                sfm_feat[cam_id] = std::vector<SFMFeature>();
            sfm_feat[cam_id].push_back(tmp_feature);
        }

        for (auto each_sfm_f : sfm_feat) {
            const auto& sensor_id = each_sfm_f.first;
            std::vector<SFMFeature>& sfm_f = each_sfm_f.second;

            std::vector<Eigen::Quaterniond> Q((int)window_frames.size(),
                                              Eigen::Quaterniond::Identity());
            std::vector<Eigen::Vector3d> T((int)window_frames.size(),
                                           Eigen::Vector3d::Zero());
            std::map<int, Eigen::Vector3d> sfm_tracked_points;

            Eigen::Matrix3d relative_R;
            Eigen::Vector3d relative_T;
            int l;
            if (!relativePose(sensor_id, relative_R, relative_T, l)) {
                std::cout
                    << "no enough features or parallax; Move device around"
                    << std::endl;
                continue;
            }

            GlobalSFM sfm;
            if (!sfm.construct(Q, T, l, relative_R, relative_T, sfm_f,
                               sfm_tracked_points)) {
                std::cout << "global SFM failed!" << std::endl;
                // marginalization_flag = MARGIN_OLD;
                continue;
            }

            {
                int push_cout = 1;
                std::list<Frame::Ptr> frame_to_imu_align;
                frame_to_imu_align.push_back(
                    Frame::Ptr(new Frame(window_frames.front())));
                frame_to_imu_align.front()->P_ = frame_to_imu_align.front()->P;
                frame_to_imu_align.front()->Q_ = frame_to_imu_align.front()->Q;
                frame_to_imu_align.front()->sensor_id = sensor_id;
                frame_to_imu_align.front()->ExT_ = ex_pose[sensor_id];
                for (int i = 1; i < window_frames.size();) {
                    Frame::Ptr new_frame(new Frame());
                    for (int j = 0; j < push_cout && i < window_frames.size();
                         j++) {
                        auto start_frame = window_frames.begin();
                        std::advance(start_frame, i);
                        Frame::Ptr curr_frame = *(start_frame);

                        new_frame->timeStamp = curr_frame->timeStamp;
                        new_frame->imuIntegrator.PushIMUMsg(
                            curr_frame->imuIntegrator.GetIMUMsg());
                        new_frame->P_ = T[i];
                        new_frame->Q_ = Q[i];
                        new_frame->sensor_id = sensor_id;
                        new_frame->ExT_ = ex_pose[sensor_id];
                        i++;
                    }
                    frame_to_imu_align.push_back(new_frame);
                }

                double scale = 1.0;
                Eigen::Vector3d GVector;
                if (tryImuAlignment(frame_to_imu_align, scale, GVector)) {
                    return 2;
                }
            }
        }
    }

    return 0;
}

bool CameraModule::relativePose(const std::string& cam_id,
                                Eigen::Matrix3d& relative_R,
                                Eigen::Vector3d& relative_T, int& l) {
    // find previous frame which contians enough correspondance and parallex
    // with newest frame

    for (int i = 0; i < window_frames.size() - 1; i++) {
        auto start_frame = window_frames.begin();
        std::advance(start_frame, i);
        auto corres =
            f_manager.getCorresponds(cam_id, i, window_frames.size() - 1);
        if (corres.size() > 20) {
            double sum_parallax = 0;
            for (int j = 0; j < int(corres.size()); j++) {
                Eigen::Vector2d pts_0(corres[j].first(0) / corres[j].first(2),
                                      corres[j].first(1) / corres[j].first(2));
                Eigen::Vector2d pts_1(
                    corres[j].second(0) / corres[j].second(2),
                    corres[j].second(1) / corres[j].second(2));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }

            double average_parallax = sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 &&
                solveRelativeRT(corres, relative_R, relative_T)) {
                l = i;
                return true;
            }
        }
    }
    return false;
}

void CameraModule::preProcess() {}

void CameraModule::postProcess() {}

void CameraModule::slideWindow(const Module::Ptr& prime_module) {}

void CameraModule::vector2double() {
    for (int i = 0; i < window_frames.size(); i++) {
        auto cf = window_frames.begin();
        std::advance(cf, i);
        Eigen::Map<Eigen::Matrix<double, 6, 1>> PR(para_Pose[i]);
        PR.segment<3>(0) = (*cf)->P;
        PR.segment<3>(3) = Sophus::SO3d((*cf)->Q).log();

        Eigen::Map<Eigen::Matrix<double, 9, 1>> VBias(para_SpeedBias[i]);
        VBias.segment<3>(0) = (*cf)->V;
        VBias.segment<3>(3) = (*cf)->bg;
        VBias.segment<3>(6) = (*cf)->ba;
    }

    for (auto c : ex_pose) {
        if (para_Ex_Pose.find(c.first) == para_Ex_Pose.end())
            para_Ex_Pose[c.first] = new double[SIZE_POSE];
        Eigen::Map<Eigen::Matrix<double, 6, 1>> Exbc(para_Ex_Pose[c.first]);
        const auto& exTbc = c.second;
        Exbc.segment<3>(0) = exTbc.block<3, 1>(0, 3);
        Exbc.segment<3>(3) =
            Sophus::SO3d(Eigen::Quaterniond(exTbc.block<3, 3>(0, 0))
                             .normalized()
                             .toRotationMatrix())
                .log();
    }

    auto dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
}

void CameraModule::addParameter() {}

void CameraModule::double2vector() {
    for (auto c : ex_pose) {
        Eigen::Map<Eigen::Matrix<double, 6, 1>> Exbc(para_Ex_Pose[c.first]);
        c.second = Eigen::Matrix4d::Identity();
        c.second.block<3, 3>(0, 0) =
            (Sophus::SO3d::exp(Exbc.segment<3>(3)).unit_quaternion())
                .toRotationMatrix();
        c.second.block<3, 1>(0, 3) = Exbc.segment<3>(0);
    }

    for (int i = 0; i < window_frames.size(); i++) {
        auto cf = window_frames.begin();
        std::advance(cf, i);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> PR(para_Pose[i]);
        Eigen::Map<const Eigen::Matrix<double, 9, 1>> VBias(para_SpeedBias[i]);
        (*cf)->P = PR.segment<3>(0);
        (*cf)->Q = Sophus::SO3d::exp(PR.segment<3>(3)).unit_quaternion();
        (*cf)->V = VBias.segment<3>(0);
        (*cf)->bg = VBias.segment<3>(3);
        (*cf)->ba = VBias.segment<3>(6);
        (*cf)->ExT_ = ex_pose[(*cf)->sensor_id];
    }

    auto dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
}

void CameraModule::addResidualBlock(int iterOpt) {
    ceres::LossFunction* loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);

    for (int i = 0; i < window_frames.size(); i++) {
        ceres::LocalParameterization* local_parameterization = NULL;
        problem->AddParameterBlock(para_Pose[i], SIZE_POSE,
                                   local_parameterization);
        problem->AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    for (auto c : para_Ex_Pose) {
        ceres::LocalParameterization* local_parameterization = NULL;
        problem->AddParameterBlock(c.second, SIZE_POSE, local_parameterization);
        if (1) {
            problem->SetParameterBlockConstant(c.second);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto& it_per_id : f_manager.feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].points[0];
        auto cam_id0 = it_per_id.feature_per_frame[0].camids[0];
        for (auto& it_per_frame : it_per_id.feature_per_frame) {
            imu_j++;
            if (imu_i != imu_j) {
                Eigen::Vector3d pts_j = it_per_frame.points[0];
                problem->AddResidualBlock(
                    ProjectionTwoFrameOneCamFactor::Create(
                        pts_i, pts_j, Eigen::Matrix2d::Identity()),
                    loss_function, para_Pose[imu_i], para_Pose[imu_j],
                    para_Ex_Pose[cam_id0], para_Feature[feature_index]);
            }

            if (it_per_frame.is_stereo) {
                for (size_t i = 1; i < it_per_frame.points.size(); i++) {
                    Eigen::Vector3d pts_j_right = it_per_frame.points[i];
                    auto cam_id1 = it_per_frame.camids[i];
                    if (imu_i != imu_j) {
                        problem->AddResidualBlock(
                            ProjectionTwoFrameTwoCamFactor::Create(
                                pts_i, pts_j_right,
                                Eigen::Matrix2d::Identity()),
                            loss_function, para_Pose[imu_i], para_Pose[imu_j],
                            para_Ex_Pose[cam_id0], para_Ex_Pose[cam_id1],
                            para_Feature[feature_index]);
                    } else {
                        problem->AddResidualBlock(
                            ProjectionOneFrameTwoCamFactor::Create(
                                pts_i, pts_j_right,
                                Eigen::Matrix2d::Identity()),
                            loss_function, para_Ex_Pose[cam_id0],
                            para_Ex_Pose[cam_id1], para_Feature[feature_index]);
                    }
                }
            }
            f_m_cnt++;
        }
    }
}

void CameraModule::marginalization1(
    MarginalizationInfo* last_marginalization_info,
    std::vector<double*>& last_marginalization_parameter_blocks,
    MarginalizationInfo* marginalization_info, int slide_win_size) {
    ceres::LossFunction* loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);

    int feature_index = -1;
    for (auto& it_per_id : f_manager.feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        if (imu_i >= slide_win_size)
            continue;

        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].points[0];
        auto cam_id0 = it_per_id.feature_per_frame[0].camids[0];

        for (auto& it_per_frame : it_per_id.feature_per_frame) {
            imu_j++;
            if (imu_i != imu_j) {
                Eigen::Vector3d pts_j = it_per_frame.points[0];
                ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
                    ProjectionTwoFrameOneCamFactor::Create(
                        pts_i, pts_j, Eigen::Matrix2d::Identity()),
                    loss_function,
                    std::vector<double*>{para_Pose[imu_i], para_Pose[imu_j],
                                         para_Ex_Pose[cam_id0],
                                         para_Feature[feature_index]},
                    std::vector<int>{0, 3});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            if (it_per_frame.is_stereo) {
                for (size_t i = 1; i < it_per_frame.points.size(); i++) {
                    Eigen::Vector3d pts_j_right = it_per_frame.points[i];
                    auto cam_id1 = it_per_frame.camids[i];
                    if (imu_i != imu_j) {
                        ResidualBlockInfo* residual_block_info =
                            new ResidualBlockInfo(
                                ProjectionTwoFrameTwoCamFactor::Create(
                                    pts_i, pts_j_right,
                                    Eigen::Matrix2d::Identity()),
                                loss_function,
                                std::vector<double*>{
                                    para_Pose[imu_i], para_Pose[imu_j],
                                    para_Ex_Pose[cam_id0],
                                    para_Ex_Pose[cam_id1],
                                    para_Feature[feature_index]},
                                std::vector<int>{0, 4});
                        marginalization_info->addResidualBlockInfo(
                            residual_block_info);
                    } else {
                        ResidualBlockInfo* residual_block_info =
                            new ResidualBlockInfo(
                                ProjectionOneFrameTwoCamFactor::Create(
                                    pts_i, pts_j_right,
                                    Eigen::Matrix2d::Identity()),
                                loss_function,
                                std::vector<double*>{
                                    para_Ex_Pose[cam_id0],
                                    para_Ex_Pose[cam_id1],
                                    para_Feature[feature_index]},
                                std::vector<int>{2});
                        marginalization_info->addResidualBlockInfo(
                            residual_block_info);
                    }
                }
            }
        }
    }
}

void CameraModule::marginalization2(
    std::unordered_map<long, double*>& addr_shift, int slide_win_size) {
    for (int i = slide_win_size; i < window_frames.size(); i++) {
        addr_shift[reinterpret_cast<long>(para_Pose[i])] =
            para_Pose[i - slide_win_size];
        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] =
            para_SpeedBias[i - slide_win_size];
    }
    for (auto c : para_Ex_Pose)
        addr_shift[reinterpret_cast<long>(c.second)] = c.second;
}

}  // namespace SensorFusion