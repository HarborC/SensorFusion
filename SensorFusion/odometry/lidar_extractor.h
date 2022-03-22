#pragma once

#include <chrono>

#include <Eigen/Core>

#include "calibration.hpp"
#include "global.h"
#include "pointcloud_viewer.h"
#include "utils/lidar_utils.h"

namespace SensorFusion {

class LidarExtractor {
public:
    using Ptr = std::shared_ptr<LidarExtractor>;
    enum Feature {
        Nor,
        Poss_Plane,
        Real_Plane,
        Edge_Jump,
        Edge_Plane,
        Wire,
        ZeroPoint
    };
    enum Surround { Prev, Next };
    enum EJump { Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind };

    struct OrgType {
        double range;
        double dista;
        double angle[2];
        double intersect;
        EJump edj[2];
        Feature ftype;
        OrgType() {
            range = 0;
            edj[Prev] = Nr_nor;
            edj[Next] = Nr_nor;
            ftype = Nor;
            intersect = 2;
        }
    };

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit LidarExtractor(const LidarCalibration &calib,
                            const bool isViz_ = false)
        : isViz(isViz_) {
        lidarType = calib.lidar_type;
        blind = calib.blind;
        pointFilterNum = calib.point_filter_num;
        SCAN_RATE = calib.scan_rate;
        N_SCAN = calib.scan_num;
        featureEnabled = calib.feature_enabled;
        givenOffsetTime = false;

        jumpUpLimit = std::cos(jumpUpLimit / 180 * M_PI);
        jumpDownLimit = std::cos(jumpDownLimit / 180 * M_PI);
        cos160 = std::cos(cos160 / 180 * M_PI);
        smallPlaneIntersect = std::cos(smallPlaneIntersect / 180 * M_PI);

        if (isViz) {
            viewer.reset(new PointCloudViewer<pcl::PointXYZRGB>());
        }
    }
    ~LidarExtractor() {}

    PointCloudType::Ptr ousterHandler(
        const pcl::PointCloud<PointOuster::Point> &pl_orig) {
        TicToc t0;
        pointCloudSurf.clear();
        pointCloudCorner.clear();

        double timeSpan = pl_orig.points.back().t;

        int plsize = pl_orig.size();
        pointCloudCorner.reserve(plsize);
        pointCloudSurf.reserve(plsize);
        if (featureEnabled) {
            for (int i = 0; i < N_SCAN; i++) {
                pointCloudBuff[i].clear();
                pointCloudBuff[i].reserve(plsize);
            }

            for (uint i = 0; i < plsize; i++) {
                double range = pl_orig.points[i].x * pl_orig.points[i].x +
                               pl_orig.points[i].y * pl_orig.points[i].y +
                               pl_orig.points[i].z * pl_orig.points[i].z;
                if (range < (blind * blind))
                    continue;
                Eigen::Vector3d pt_vec;
                PointType added_pt;
                added_pt.x = pl_orig.points[i].x;
                added_pt.y = pl_orig.points[i].y;
                added_pt.z = pl_orig.points[i].z;
                added_pt.intensity = pl_orig.points[i].intensity;
                added_pt.normal_x = pl_orig.points[i].t / timeSpan;
                added_pt.normal_y = (float)pl_orig.points[i].ring;
                added_pt.normal_z = 0.0;
                double yaw_angle = RAD2DEG(std::atan2(added_pt.y, added_pt.x));
                if (yaw_angle >= 180.0)
                    yaw_angle -= 360.0;
                if (yaw_angle <= -180.0)
                    yaw_angle += 360.0;

                if (pl_orig.points[i].ring < N_SCAN) {
                    pointCloudBuff[pl_orig.points[i].ring].push_back(added_pt);
                }
            }

            for (int j = 0; j < N_SCAN; j++) {
                PointCloudType &pl = pointCloudBuff[j];
                int linesize = pl.size();
                std::vector<OrgType> &types = pointTypes[j];
                types.clear();
                types.resize(linesize);
                linesize--;
                for (uint i = 0; i < linesize; i++) {
                    types[i].range =
                        sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
                    vx = pl[i].x - pl[i + 1].x;
                    vy = pl[i].y - pl[i + 1].y;
                    vz = pl[i].z - pl[i + 1].z;
                    types[i].dista = vx * vx + vy * vy + vz * vz;
                }
                types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x +
                                             pl[linesize].y * pl[linesize].y);
                giveFeature(pl, types);
            }
        } else {
            for (int i = 0; i < pl_orig.points.size(); i++) {
                if (i % pointFilterNum != 0)
                    continue;

                double range = pl_orig.points[i].x * pl_orig.points[i].x +
                               pl_orig.points[i].y * pl_orig.points[i].y +
                               pl_orig.points[i].z * pl_orig.points[i].z;

                if (range < (blind * blind))
                    continue;

                Eigen::Vector3d pt_vec;
                PointType added_pt;
                added_pt.x = pl_orig.points[i].x;
                added_pt.y = pl_orig.points[i].y;
                added_pt.z = pl_orig.points[i].z;
                added_pt.intensity = pl_orig.points[i].intensity;
                added_pt.normal_x = pl_orig.points[i].t / timeSpan;
                added_pt.normal_y = (float)pl_orig.points[i].ring;
                added_pt.normal_z = 2.0;

                pointCloudSurf.points.push_back(added_pt);
            }
        }

        PointCloudType::Ptr result(new PointCloudType);
        for (int i = 0; i < pointCloudSurf.points.size(); i++) {
            pointCloudSurf.points[i].normal_z = 2.0;
            result->points.push_back(pointCloudSurf.points[i]);
        }

        for (int i = 0; i < pointCloudCorner.points.size(); i++) {
            pointCloudCorner.points[i].normal_z = 1.0;
            result->points.push_back(pointCloudCorner.points[i]);
        }

        if (isViz)
            viewer->addPointCloud(utility::convertToRGB(*result),
                                  "feat_points");

        std::string remark = "get features";
        logger->recordLogger(logger_flag, t0.toc(), frame_counter, remark);

        frame_counter++;

        return result;
    }

    PointCloudType::Ptr velodyneHandler(
        const pcl::PointCloud<PointVelodyne::Point> &pl_orig) {
        TicToc t0;

        pointCloudSurf.clear();
        pointCloudCorner.clear();

        int plsize = pl_orig.points.size();
        pointCloudSurf.reserve(plsize);

        /*** These variables only works when no point timestamps given ***/
        double timeSpan = 0.0;
        double omega_l = 0.361 * SCAN_RATE;  // scan angular velocity
        std::vector<bool> is_first(N_SCAN, true);
        std::vector<double> yaw_fp(N_SCAN, 0.0);    // yaw of first scan point
        std::vector<float> time_last(N_SCAN, 0.0);  // last offset time
        /*****************************************************************/

        if (pl_orig.points[plsize - 1].time > 0) {
            givenOffsetTime = true;
        } else {
            givenOffsetTime = false;
        }

        if (featureEnabled) {
            for (int i = 0; i < N_SCAN; i++) {
                pointCloudBuff[i].clear();
                pointCloudBuff[i].reserve(plsize);
            }

            for (int i = 0; i < plsize; i++) {
                int layer = pl_orig.points[i].ring;
                if (layer >= N_SCAN)
                    continue;
                PointType added_pt;
                added_pt.normal_x = 0;
                added_pt.normal_y = (float)pl_orig.points[i].ring;
                added_pt.normal_z = 0;
                added_pt.x = pl_orig.points[i].x;
                added_pt.y = pl_orig.points[i].y;
                added_pt.z = pl_orig.points[i].z;
                added_pt.intensity = pl_orig.points[i].intensity;

                if (!givenOffsetTime) {
                    double yaw_angle =
                        RAD2DEG(std::atan2(added_pt.y, added_pt.x));
                    if (is_first[layer]) {
                        yaw_fp[layer] = yaw_angle;
                        is_first[layer] = false;
                        time_last[layer] = 0.0;
                        added_pt.normal_x = 0.0;
                        continue;
                    }

                    if (yaw_angle <= yaw_fp[layer]) {
                        added_pt.normal_x =
                            (yaw_fp[layer] - yaw_angle) / omega_l;
                    } else {
                        added_pt.normal_x =
                            (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
                    }

                    if (added_pt.normal_x < time_last[layer])
                        added_pt.normal_x += 360.0 / omega_l;

                    time_last[layer] = added_pt.normal_x;
                } else {
                    added_pt.normal_x = pl_orig.points[i].time;
                }

                if (added_pt.normal_x > timeSpan)
                    timeSpan = added_pt.normal_x;

                pointCloudBuff[layer].points.push_back(added_pt);
            }

            for (int j = 0; j < N_SCAN; j++) {
                PointCloudType &pl = pointCloudBuff[j];
                for (auto &p : pl.points) {
                    p.normal_x = p.normal_x / timeSpan;
                }
                int linesize = pl.size();
                if (linesize < 2)
                    continue;
                std::vector<OrgType> &types = pointTypes[j];
                types.clear();
                types.resize(linesize);
                linesize--;
                for (uint i = 0; i < linesize; i++) {
                    types[i].range =
                        sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
                    vx = pl[i].x - pl[i + 1].x;
                    vy = pl[i].y - pl[i + 1].y;
                    vz = pl[i].z - pl[i + 1].z;
                    types[i].dista = vx * vx + vy * vy + vz * vz;
                }
                types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x +
                                             pl[linesize].y * pl[linesize].y);
                giveFeature(pl, types);
            }
        } else {
            for (int i = 0; i < plsize; i++) {
                PointType added_pt;

                added_pt.normal_x = 0;
                added_pt.normal_y = (float)pl_orig.points[i].ring;
                added_pt.normal_z = 2.0;
                added_pt.x = pl_orig.points[i].x;
                added_pt.y = pl_orig.points[i].y;
                added_pt.z = pl_orig.points[i].z;
                added_pt.intensity = pl_orig.points[i].intensity;

                if (!givenOffsetTime) {
                    int layer = pl_orig.points[i].ring;
                    double yaw_angle =
                        RAD2DEG(std::atan2(added_pt.y, added_pt.x));

                    if (is_first[layer]) {
                        yaw_fp[layer] = yaw_angle;
                        is_first[layer] = false;
                        time_last[layer] = 0.0;
                        added_pt.normal_x = 0.0;
                        continue;
                    }

                    // compute offset time
                    if (yaw_angle <= yaw_fp[layer]) {
                        added_pt.normal_x =
                            (yaw_fp[layer] - yaw_angle) / omega_l;
                    } else {
                        added_pt.normal_x =
                            (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
                    }

                    if (added_pt.normal_x < time_last[layer])
                        added_pt.normal_x += 360.0 / omega_l;

                    time_last[layer] = added_pt.normal_x;
                } else {
                    added_pt.normal_x = pl_orig.points[i].time;
                }

                if (added_pt.normal_x > timeSpan)
                    timeSpan = added_pt.normal_x;

                if (i % pointFilterNum == 0) {
                    if (added_pt.x * added_pt.x + added_pt.y * added_pt.y +
                            added_pt.z * added_pt.z >
                        (blind * blind)) {
                        pointCloudSurf.points.push_back(added_pt);
                    }
                }
            }

            for (int i = 0; i < pointCloudSurf.points.size(); i++) {
                pointCloudSurf.points[i].normal_x =
                    pointCloudSurf.points[i].normal_x / timeSpan;
            }
        }

        PointCloudType::Ptr result(new PointCloudType);
        for (int i = 0; i < pointCloudSurf.points.size(); i++) {
            pointCloudSurf.points[i].normal_z = 2.0;
            result->points.push_back(pointCloudSurf.points[i]);
        }

        for (int i = 0; i < pointCloudCorner.points.size(); i++) {
            pointCloudCorner.points[i].normal_z = 1.0;
            result->points.push_back(pointCloudCorner.points[i]);
        }

        if (isViz)
            viewer->addPointCloud(utility::convertToRGB(*result),
                                  "feat_points");

        std::string remark = "get features";
        logger->recordLogger(logger_flag, t0.toc(), frame_counter, remark);

        frame_counter++;

        return result;
    }

    void giveFeature(pcl::PointCloud<PointType> &pl,
                     std::vector<OrgType> &types) {
        int plsize = pl.size();
        int plsize2;
        if (plsize == 0) {
            printf("something wrong\n");
            return;
        }
        uint head = 0;

        while (types[head].range < blind) {
            head++;
        }

        // Surf
        plsize2 = (plsize > groupSize) ? (plsize - groupSize) : 0;

        Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());
        Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());

        uint i_nex = 0, i2;
        uint last_i = 0;
        uint last_i_nex = 0;
        int last_state = 0;
        int plane_type;

        for (uint i = head; i < plsize2; i++) {
            if (types[i].range < blind) {
                continue;
            }

            i2 = i;

            plane_type = planeJudge(pl, types, i, i_nex, curr_direct);

            if (plane_type == 1) {
                for (uint j = i; j <= i_nex; j++) {
                    if (j != i && j != i_nex) {
                        types[j].ftype = Real_Plane;
                    } else {
                        types[j].ftype = Poss_Plane;
                    }
                }

                // if(last_state==1 && fabs(last_direct.sum())>0.5)
                if (last_state == 1 && last_direct.norm() > 0.1) {
                    double mod = last_direct.transpose() * curr_direct;
                    if (mod > -0.707 && mod < 0.707) {
                        types[i].ftype = Edge_Plane;
                    } else {
                        types[i].ftype = Real_Plane;
                    }
                }

                i = i_nex - 1;
                last_state = 1;
            } else {
                i = i_nex;
                last_state = 0;
            }

            last_i = i2;
            last_i_nex = i_nex;
            last_direct = curr_direct;
        }

        plsize2 = plsize > 3 ? plsize - 3 : 0;
        for (uint i = head + 3; i < plsize2; i++) {
            if (types[i].range < blind || types[i].ftype >= Real_Plane) {
                continue;
            }

            if (types[i - 1].dista < 1e-16 || types[i].dista < 1e-16) {
                continue;
            }

            Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);
            Eigen::Vector3d vecs[2];

            for (int j = 0; j < 2; j++) {
                int m = -1;
                if (j == 1) {
                    m = 1;
                }

                if (types[i + m].range < blind) {
                    if (types[i].range > infBound) {
                        types[i].edj[j] = Nr_inf;
                    } else {
                        types[i].edj[j] = Nr_blind;
                    }
                    continue;
                }

                vecs[j] =
                    Eigen::Vector3d(pl[i + m].x, pl[i + m].y, pl[i + m].z);
                vecs[j] = vecs[j] - vec_a;

                types[i].angle[j] =
                    vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm();
                if (types[i].angle[j] < jumpUpLimit) {
                    types[i].edj[j] = Nr_180;
                } else if (types[i].angle[j] > jumpDownLimit) {
                    types[i].edj[j] = Nr_zero;
                }
            }

            types[i].intersect = vecs[Prev].dot(vecs[Next]) /
                                 vecs[Prev].norm() / vecs[Next].norm();
            if (types[i].edj[Prev] == Nr_nor && types[i].edj[Next] == Nr_zero &&
                types[i].dista > 0.0225 &&
                types[i].dista > 4 * types[i - 1].dista) {
                if (types[i].intersect > cos160) {
                    if (edgeJumpJudge(pl, types, i, Prev)) {
                        types[i].ftype = Edge_Jump;
                    }
                }
            } else if (types[i].edj[Prev] == Nr_zero &&
                       types[i].edj[Next] == Nr_nor &&
                       types[i - 1].dista > 0.0225 &&
                       types[i - 1].dista > 4 * types[i].dista) {
                if (types[i].intersect > cos160) {
                    if (edgeJumpJudge(pl, types, i, Next)) {
                        types[i].ftype = Edge_Jump;
                    }
                }
            } else if (types[i].edj[Prev] == Nr_nor &&
                       types[i].edj[Next] == Nr_inf) {
                if (edgeJumpJudge(pl, types, i, Prev)) {
                    types[i].ftype = Edge_Jump;
                }
            } else if (types[i].edj[Prev] == Nr_inf &&
                       types[i].edj[Next] == Nr_nor) {
                if (edgeJumpJudge(pl, types, i, Next)) {
                    types[i].ftype = Edge_Jump;
                }

            } else if (types[i].edj[Prev] > Nr_nor &&
                       types[i].edj[Next] > Nr_nor) {
                if (types[i].ftype == Nor) {
                    types[i].ftype = Wire;
                }
            }
        }

        plsize2 = plsize - 1;
        double ratio;
        for (uint i = head + 1; i < plsize2; i++) {
            if (types[i].range < blind || types[i - 1].range < blind ||
                types[i + 1].range < blind) {
                continue;
            }

            if (types[i - 1].dista < 1e-8 || types[i].dista < 1e-8) {
                continue;
            }

            if (types[i].ftype == Nor) {
                if (types[i - 1].dista > types[i].dista) {
                    ratio = types[i - 1].dista / types[i].dista;
                } else {
                    ratio = types[i].dista / types[i - 1].dista;
                }

                if (types[i].intersect < smallPlaneIntersect &&
                    ratio < smallPlaneRatio) {
                    if (types[i - 1].ftype == Nor) {
                        types[i - 1].ftype = Real_Plane;
                    }
                    if (types[i + 1].ftype == Nor) {
                        types[i + 1].ftype = Real_Plane;
                    }
                    types[i].ftype = Real_Plane;
                }
            }
        }

        int last_surface = -1;
        for (uint j = head; j < plsize; j++) {
            if (types[j].ftype == Poss_Plane || types[j].ftype == Real_Plane) {
                if (last_surface == -1) {
                    last_surface = j;
                }

                if (j == uint(last_surface + pointFilterNum - 1)) {
                    pointCloudSurf.push_back(pl[j]);

                    last_surface = -1;
                }
            } else {
                if (types[j].ftype == Edge_Jump ||
                    types[j].ftype == Edge_Plane) {
                    pointCloudCorner.push_back(pl[j]);
                }
                if (last_surface != -1) {
                    PointType ap;
                    for (uint k = last_surface; k < j; k++) {
                        ap.x += pl[k].x;
                        ap.y += pl[k].y;
                        ap.z += pl[k].z;
                        ap.intensity += pl[k].intensity;
                        ap.normal_x += pl[k].normal_x;
                        ap.normal_y += pl[k].normal_y;
                    }
                    ap.x /= (j - last_surface);
                    ap.y /= (j - last_surface);
                    ap.z /= (j - last_surface);
                    ap.intensity /= (j - last_surface);
                    ap.normal_x /= (j - last_surface);
                    ap.normal_y /= (j - last_surface);
                    pointCloudSurf.push_back(ap);
                }
                last_surface = -1;
            }
        }
    }

    int planeJudge(const PointCloudType &pl, std::vector<OrgType> &types,
                   uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct) {
        double group_dis = disA * types[i_cur].range + disB;
        group_dis = group_dis * group_dis;
        // i_nex = i_cur;

        double two_dis;
        std::vector<double> disarr;
        disarr.reserve(20);

        for (i_nex = i_cur; i_nex < i_cur + groupSize; i_nex++) {
            if (types[i_nex].range < blind) {
                curr_direct.setZero();
                return 2;
            }
            disarr.push_back(types[i_nex].dista);
        }

        for (;;) {
            if ((i_cur >= pl.size()) || (i_nex >= pl.size()))
                break;

            if (types[i_nex].range < blind) {
                curr_direct.setZero();
                return 2;
            }
            vx = pl[i_nex].x - pl[i_cur].x;
            vy = pl[i_nex].y - pl[i_cur].y;
            vz = pl[i_nex].z - pl[i_cur].z;
            two_dis = vx * vx + vy * vy + vz * vz;
            if (two_dis >= group_dis) {
                break;
            }
            disarr.push_back(types[i_nex].dista);
            i_nex++;
        }

        double leng_wid = 0;
        double v1[3], v2[3];
        for (uint j = i_cur + 1; j < i_nex; j++) {
            if ((j >= pl.size()) || (i_cur >= pl.size()))
                break;
            v1[0] = pl[j].x - pl[i_cur].x;
            v1[1] = pl[j].y - pl[i_cur].y;
            v1[2] = pl[j].z - pl[i_cur].z;

            v2[0] = v1[1] * vz - vy * v1[2];
            v2[1] = v1[2] * vx - v1[0] * vz;
            v2[2] = v1[0] * vy - vx * v1[1];

            double lw = v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2];
            if (lw > leng_wid) {
                leng_wid = lw;
            }
        }

        if ((two_dis * two_dis / leng_wid) < p2lRatio) {
            curr_direct.setZero();
            return 0;
        }

        uint disarrsize = disarr.size();
        for (uint j = 0; j < disarrsize - 1; j++) {
            for (uint k = j + 1; k < disarrsize; k++) {
                if (disarr[j] < disarr[k]) {
                    leng_wid = disarr[j];
                    disarr[j] = disarr[k];
                    disarr[k] = leng_wid;
                }
            }
        }

        if (disarr[disarr.size() - 2] < 1e-16) {
            curr_direct.setZero();
            return 0;
        }

        {
            double dismax_min = disarr[0] / disarr[disarrsize - 2];
            if (dismax_min >= limitMaxMin) {
                curr_direct.setZero();
                return 0;
            }
        }

        curr_direct << vx, vy, vz;
        curr_direct.normalize();
        return 1;
    }

    bool edgeJumpJudge(const PointCloudType &pl, std::vector<OrgType> &types,
                       uint i, Surround nor_dir) {
        if (nor_dir == 0) {
            if (types[i - 1].range < blind || types[i - 2].range < blind) {
                return false;
            }
        } else if (nor_dir == 1) {
            if (types[i + 1].range < blind || types[i + 2].range < blind) {
                return false;
            }
        }
        double d1 = types[i + nor_dir - 1].dista;
        double d2 = types[i + 3 * nor_dir - 2].dista;
        double d;

        if (d1 < d2) {
            d = d1;
            d1 = d2;
            d2 = d;
        }

        d1 = sqrt(d1);
        d2 = sqrt(d2);

        if (d1 > edgeA * d2 || (d1 - d2) > edgeB) {
            return false;
        }

        return true;
    }

    PointCloudType pointCloudCorner, pointCloudSurf;
    PointCloudType pointCloudBuff[128];    // maximum 128 line
    std::vector<OrgType> pointTypes[128];  // maximum 128 line
    bool givenOffsetTime = false;          // give OffsetTime
    bool featureEnabled = false;
    double blind = 1.0;
    int pointFilterNum = 1;
    int SCAN_RATE = 10;
    int N_SCAN = 16;
    int lidarType = LidarType::VELODYNE;

    // params for extractor
    double vx, vy, vz;

    double infBound = 10;
    int groupSize = 8;
    double disA = 0.01;
    double disB = 0.1;
    double p2lRatio = 225;
    double limitMaxMid = 6.25;
    double limitMidMin = 6.25;
    double limitMaxMin = 3.24;
    double jumpUpLimit = 170.0;
    double jumpDownLimit = 8.0;
    double cos160 = 160.0;
    double edgeA = 2;
    double edgeB = 0.1;
    double smallPlaneIntersect = 172.5;
    double smallPlaneRatio = 1.2;

    bool isViz;
    PointCloudViewer<pcl::PointXYZRGB>::Ptr viewer;

    std::string logger_flag = "!lidar_extractor! : ";
    int frame_counter = 0;
};
}  // namespace SensorFusion