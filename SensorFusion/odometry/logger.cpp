/*
 * @Author: Jiagang Chen
 * @Date: 2021-12-09 12:29:38
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-12-10 03:55:13
 * @Description: ...
 * @Reference: ...
 */

#include "logger.h"

#include <iomanip>
#include <sstream>

Logger::Logger(const std::string &_log_path) : log_path(_log_path) {
    this->ost_record_invaild_image.open(this->log_path + "/invaild_image.txt",
                                        std::ios::trunc);

    this->ost_record_odometry_time.open(this->log_path + "/odometry_time.txt",
                                        std::ios::trunc);
    this->ost_record_logger.open(this->log_path + "/logger.txt",
                                 std::ios::trunc);
}
Logger::~Logger() {
    this->ost_record_invaild_image.close();
    this->ost_record_odometry_time.close();
    this->ost_record_logger.close();
}

void Logger::recordInvaildImage(int index, double timestamp, std::string path) {
    this->ost_record_invaild_image << std::fixed << index << " "
                                   << std::setprecision(8) << timestamp << " "
                                   << path << std::endl;
}

void Logger::recordOdometryTime(double timestamp,
                                std::vector<double> speedtimes) {
    this->ost_record_odometry_time << "image timestamp: " << std::fixed
                                   << std::setprecision(8) << timestamp << " "
                                   << std::endl
                                   << "[" << std::endl;
    this->ost_record_odometry_time << "  getImage time is " << std::fixed
                                   << std::setprecision(8) << speedtimes[0]
                                   << " ms" << std::endl;
    this->ost_record_odometry_time << "  match time is " << std::fixed
                                   << std::setprecision(8) << speedtimes[1]
                                   << " ms" << std::endl;

    this->ost_record_odometry_time << "]" << std::endl << std::endl;
}

void Logger::recordLogger(const std::string &flag, const double &time,
                          const int &index, const std::string &line) {
    std::unique_lock<std::mutex> locker(mt_record_logger);
    this->ost_record_logger << flag << " [ " << std::fixed
                            << std::setprecision(8) << time << "(ms) ] "
                            << " ( " << index << " : " << line << " ) "
                            << std::endl;
    locker.unlock();
}