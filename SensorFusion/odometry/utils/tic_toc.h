/*
 * @Author: Jiagang Chen
 * @Date: 2021-12-09 17:23:14
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-12-09 17:23:14
 * @Description: ...
 * @Reference: ...
 */
#pragma once

#include <chrono>
#include <cstdlib>
#include <ctime>

class TicToc {
public:
    TicToc() { tic(); }

    void tic() { start = std::chrono::system_clock::now(); }

    double toc() {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
