#ifndef _SENSOR_FUSION_DATABASE_H_
#define _SENSOR_FUSION_DATABASE_H_

#include "../Sensors/Imu/imu.h"
#include "../common_header.h"

namespace SensorFusion {

class DataBase {
public:
    typedef std::shared_ptr<DataBase> Ptr;
    DataBase();
    ~DataBase() {}

    bool feed_imu_data(const DatasetIO::AccelData::Ptr& accel,
                       const DatasetIO::GyroData::Ptr& gyro);

    bool feed_wheel_data(const DatasetIO::WheelData::Ptr& wheel){};

    bool feed_mono_image_data(const DatasetIO::MonoImageData::Ptr& mono_image);

    bool feed_gnss_data(const DatasetIO::GnssData::Ptr& gnss);

    void feed_groundtruth(const DatasetIO::PoseData::Ptr& gt_pose){};

    bool get_sync_data(
        std::vector<std::vector<DatasetIO::Measurement::Ptr>>& data);

    void print();

public:
    int max_imu_buffer_length_ = 1000;
    std::deque<DatasetIO::ImuData::Ptr> imu_buffer_;

    int max_wheel_buffer_length_ = 1000;
    std::deque<DatasetIO::WheelData::Ptr> wheel_buffer_;

    int max_gnss_buffer_length_ = 100;
    std::deque<DatasetIO::GnssData::Ptr> gnss_buffer_;

    int max_mono_image_buffer_length_ = 10;
    std::deque<DatasetIO::MonoImageData::Ptr> mono_image_buffer_;

protected:
    std::mutex m_data_mutex;
    std::mutex m_imu_mutex;
    std::mutex m_wheel_mutex;
    std::mutex m_gnss_mutex;
    std::mutex m_mono_image_mutex;

    double last_timestamp_;
};

}  // namespace SensorFusion

#endif