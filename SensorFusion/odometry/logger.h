/*
 * @Author: Jiagang Chen
 * @Date: 2021-12-09 12:29:38
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-12-10 03:55:13
 * @Description: ...
 * @Reference: ...
 */

#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "utils/tic_toc.h"

class Logger {
public:
    typedef std::shared_ptr<Logger> Ptr;
    Logger() {}
    Logger(const std::string &_log_path);
    ~Logger();

    void recordInvaildImage(int index, double timestamp, std::string path);

    void recordOdometryTime(double timestamp, std::vector<double> speedtimes);

    void recordLogger(const std::string &flag, const double &time,
                      const int &index, const std::string &line);

    std::string log_path;

protected:
    std::ofstream ost_record_invaild_image;
    std::ofstream ost_record_odometry_time;
    std::ofstream ost_record_logger;

    std::mutex mt_record_odometry_time;
    std::mutex mt_record_logger;
};