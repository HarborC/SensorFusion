/*
 * @Author: Jiagang Chen
 * @Date: 2021-11-05 02:04:56
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-05 02:09:22
 * @Description: ...
 * @Reference: ...
 */
#ifndef _UMMLT_LIVO_SLAM_SYSTEM_
#define _UMMLT_LIVO_SLAM_SYSTEM_

#include <iostream>
#include <string>

namespace ummlt_livo {

class SLAMSystem {
public:
    SLAMSystem(const std::string &config_path) : config_path_(config_path) {};
    ~SLAMSystem() {};

    void run();

public:
    std::string config_path_;

};

}

#endif