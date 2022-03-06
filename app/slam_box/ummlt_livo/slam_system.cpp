/*
 * @Author: Jiagang Chen
 * @Date: 2021-11-05 02:10:36
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-05 04:04:27
 * @Description: ...
 * @Reference: ...
 */
#include "slam_system.h"

namespace ummlt_livo {

void SLAMSystem::run() {

// 读取配置文件配置整个系统

// 循环处理数据
while (1) {
    // 获取下一个数据
    /*get_next_data();*/

    // VIO
    if (/*is camera data*/) { // 如果是相机数据 

        // 跟踪前一帧影像的特征点

        // 获取两帧影像间的所有的IMU观测值，计算预积分

        if (!/*is initialized*/) {
            /*initialize()*/
        }

        // 使用IMU观测值直接预测当前帧影像的位置和姿态， 或者加入滤波

        // 进行特征点的三角化

        // 进行滑动窗口的优化
        
    }

    // LIO
    if (/*is lidar data*/) {

    }

    // 如果满足某一条件退出循环
    if (/*condition*/) {
        break;
    }
}



}

}