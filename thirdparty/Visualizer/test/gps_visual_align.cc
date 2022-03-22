#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_set>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;

namespace utils {

template <class T>
inline std::vector<T> find_closest_elements(const std::vector<T> &arr,
                                            const size_t &k, const T &x) {
    if (arr.size() < k || arr.size() == 0 || k <= 0)
        return std::vector<T>();

    size_t l = 0;
    size_t r = arr.size() - k;
    while (l < r) {
        size_t m = (l + r) / 2;
        if (x - arr[m] > arr[m + k] - x)
            l = m + 1;
        else
            r = m;
    }

    return std::vector<T>(arr.begin() + l, arr.begin() + k + l);
}

template <class T>
inline std::vector<size_t> find_closest_index(const std::vector<T> &arr,
                                              const size_t &k, const T &x) {
    if (arr.size() < k || arr.size() == 0 || k <= 0)
        return std::vector<size_t>();

    size_t l = 0;
    size_t r = arr.size() - k;
    while (l < r) {
        size_t m = (l + r) / 2;
        if (x - arr[m] > arr[m + k] - x)
            l = m + 1;
        else
            r = m;
    }

    std::vector<size_t> index;
    for (size_t i = 0; i < k; i++) {
        index.push_back(l + i);
    }

    return index;
}

}

struct GtData {
    double ts;
    Eigen::Vector3d trans;

    static std::vector<double> get_vector_ts(const std::vector<GtData> &gt) {
        std::vector<double> all_ts;
        for (int i = 0; i < gt.size(); i++) {
            all_ts.push_back(gt[i].ts);
        }
        return all_ts;
    }
};

struct OdomData {
    double ts;
    Eigen::Vector3d trans;
    Eigen::Matrix3d rot;

    static std::vector<double> get_vector_ts(const std::vector<OdomData> &odom) {
        std::vector<double> all_ts;
        for (int i = 0; i < odom.size(); i++) {
            all_ts.push_back(odom[i].ts);
        }
        return all_ts;
    }
};

void read_gt(const std::string &path, std::vector<GtData> &gt) {
    std::ifstream f_read;
    f_read.open(path.c_str());

    while (!f_read.eof()) {
        std::string s;
        std::getline(f_read, s);
        if (s[0] == '#' || s == "")
            continue;

        if (!s.empty()) {
            std::string item;
            size_t trans = 0;
            double data_item[8];
            int count = 0;
            while ((trans = s.find(' ')) != std::string::npos) {
                item = s.substr(0, trans);
                data_item[count++] = std::stod(item);
                s.erase(0, trans + 1);
            }
            item = s.substr(0, trans);
            data_item[7] = std::stod(item);

            GtData gt_item;
            gt_item.ts = data_item[0] * 1e-9;
            gt_item.trans = Eigen::Vector3d(data_item[1], data_item[2], data_item[3]);
            gt.push_back(gt_item);
        }
    }
}

void read_odom(const std::string &path, std::vector<OdomData> &odom) {
    std::ifstream f_read;
    f_read.open(path.c_str());

    while (!f_read.eof()) {
        std::string s;
        std::getline(f_read, s);
        if (s[0] == '#' || s == "")
            continue;

        if (!s.empty()) {
            std::string item;
            size_t trans = 0;
            double data_item[8];
            int count = 0;
            while ((trans = s.find(' ')) != std::string::npos) {
                item = s.substr(0, trans);
                data_item[count++] = std::stod(item);
                s.erase(0, trans + 1);
            }
            item = s.substr(0, trans);
            data_item[7] = std::stod(item);

            OdomData odom_item;
            odom_item.ts = data_item[0] * 1e-9;
            odom_item.trans = Eigen::Vector3d(data_item[1], data_item[2], data_item[3]);
            odom_item.rot = Eigen::Quaterniond(data_item[7], data_item[4], data_item[5], data_item[6]).toRotationMatrix();
            odom.push_back(odom_item);
        }
    }
}

int search_align_gt_odom(const int &cur_odom_index, const double &offset_ts, const std::vector<GtData> &gt, std::vector<int> &align_index_gt, const std::vector<OdomData> &odom, std::vector<int> &align_index_odom) {

    double curr_odom_ts = old_odom[cur_odom_index].ts;
    auto all_gt_ts = GtData::get_vector_ts(gt);
    auto all_odom_ts = OdomData::get_vector_ts(odom);

    auto cls_idx = find_closest_index(all_gt_ts, 1, curr_odom_ts + offset_ts);
    int local_idx_in_gt_index = cls_idx[i];



    return local_idx_in_odom_index;
}

void align(const std::vector<GtData> &gt, const std::vector<OdomData> &old_odom, const double &offset_ts, std::vector<OdomData> &new_odom) {
    for (int i = 0; i < old_odom.size(); i++) {
        std::vector<int> align_index_gt;
        std::vector<int> align_index_odom;
        int local_idx_in_odom_index = search_align_gt_odom(i, offset_ts, gt, align_index_gt, old_odom, align_index_odom);

        OdomData curr_odom = sim3_align_odom(curr_odom_index, gt, align_index_gt, old_odom, align_index_odom);
        new_odom.push_back(curr_odom);
    }
}

void save_odom(const std::string &save_path, const std::vector<OdomData> &odom) {
    std::ofstream out(save_path);
    for (int i = 0; i < odom.size(); i++) {
        out << std::fixed << std::setprecesion(0) << odom[i].ts * 1e9 << " ";
        out << std::fixed << std::setprecesion(12) << odom[i].trans(0) << " " << odom[i].trans(1) << " " << odom[i].trans(2) << " ";
        Eigen::Quaterniond q_rot(odom[i].rot);
        out << std::fixed << std::setprecesion(12) << q_rot.x() << " " << q_rot.y() << " " << q_rot.z() << " " << q_rot.w() << std::endl; 
    }
    out.close();
}

int main(int argc, const char** argv) {

    std::vector<GtData> gt;
    read_gt(std::string(argv[1]), gt);

    std::vector<OdomData> odom;
    read_odom(std::string(argv[2], odom);

    double offset = std::stod(std::string(argv[4]));

    std::vector<OdomData> new_odom;
    align(gt, odom, offset, new_odom);

    save_odom(std::string(argv[3]), new_odom);

    return 0;
}