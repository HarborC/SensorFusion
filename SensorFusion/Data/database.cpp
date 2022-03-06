#include "database.h"

namespace SensorFusion {

DataBase::DataBase() {}

bool DataBase::feed_imu_data(const DatasetIO::AccelData::Ptr& accel_data,
                             const DatasetIO::GyroData::Ptr& gyro_data) {
    unique_lock<mutex> lock(m_imu_mutex);

    DatasetIO::ImuData::Ptr imu_data(new DatasetIO::ImuData());
    imu_data->timestamp = accel_data->timestamp;
    imu_data->am = accel_data->data;
    imu_data->wm = gyro_data->data;

    // Reject rolling back wheel data.
    if (!imu_buffer_.empty() &&
        imu_data->timestamp <= imu_buffer_.back()->timestamp) {
        LOG(ERROR) << "imu timestamp rolling back!";
        return false;
    }

    // LOG(INFO) << "get one imu data";

    // First push data to deque.
    imu_buffer_.push_back(imu_data);
    if (imu_buffer_.size() > max_imu_buffer_length_) {
        imu_buffer_.pop_front();
    }

    return true;
}

bool DataBase::feed_mono_image_data(
    const DatasetIO::MonoImageData::Ptr& image_data) {
    unique_lock<mutex> lock(m_mono_image_mutex);

    // Reject rolling back camera data.
    if (!mono_image_buffer_.empty() &&
        image_data->timestamp <= mono_image_buffer_.back()->timestamp) {
        LOG(ERROR) << "Image timestamp rolling back!";
        return false;
    }

    // LOG(INFO) << "get one mono image data";

    mono_image_buffer_.push_back(image_data);
    while (mono_image_buffer_.size() > max_mono_image_buffer_length_ ||
           (!wheel_buffer_.empty() && mono_image_buffer_.front()->timestamp <
                                          wheel_buffer_.front()->timestamp) ||
           (!imu_buffer_.empty() && mono_image_buffer_.front()->timestamp <
                                        imu_buffer_.front()->timestamp)) {
        mono_image_buffer_.pop_front();
    }

    return true;
}

bool DataBase::feed_gnss_data(const DatasetIO::GnssData::Ptr& gnss_data) {
    unique_lock<mutex> lock(m_gnss_mutex);

    // Reject rolling back gnss data.
    if (!gnss_buffer_.empty() &&
        gnss_data->timestamp <= gnss_buffer_.back()->timestamp) {
        LOG(ERROR) << "Gnss timestamp rolling back!";
        return false;
    }

    // LOG(INFO) << "get one gnss data";

    gnss_buffer_.push_back(gnss_data);
    while (gnss_buffer_.size() > max_gnss_buffer_length_ ||
           (!wheel_buffer_.empty() && gnss_buffer_.front()->timestamp <
                                          wheel_buffer_.front()->timestamp) ||
           (!imu_buffer_.empty() &&
            gnss_buffer_.front()->timestamp < imu_buffer_.front()->timestamp)) {
        gnss_buffer_.pop_front();
    }

    return true;
}

bool DataBase::get_sync_data(
    std::vector<std::vector<DatasetIO::Measurement::Ptr>>& data) {
    data.clear();

    if (mono_image_buffer_.empty() && gnss_buffer_.empty()) {
        return false;
    }

    const double mono_image_timestamp = mono_image_buffer_.front()->timestamp;
    const double gnss_timestamp = gnss_buffer_.front()->timestamp;

    if (imu_buffer_.back()->timestamp < mono_image_timestamp) {
        return false;
    }

    auto mono_image_data = mono_image_buffer_.front();
    mono_image_buffer_.pop_front();

    // Collect all wheel data before the first image.
    if (last_timestamp_ <= 0.) {
        last_timestamp_ = imu_buffer_.front()->timestamp;
    }

    // Collect Wheel data between the last timestamp and the current image time.
    std::vector<DatasetIO::ImuData::Ptr> imu_data_segment =
        select_imu_readings(imu_buffer_, last_timestamp_, mono_image_timestamp);
    if (imu_data_segment.size() < 2) {
        LOG(ERROR) << "Failed to collect imu data between times!";
        return false;
    }

    std::vector<DatasetIO::Measurement::Ptr> imu_data_segment_temp;
    for (int i = 0; i < imu_data_segment.size(); i++) {
        imu_data_segment_temp.push_back(imu_data_segment[i]);
    }
    data.push_back(imu_data_segment_temp);

    std::vector<DatasetIO::Measurement::Ptr> mono_image_data_segment_temp;
    mono_image_data_segment_temp.push_back(mono_image_data);
    data.push_back(mono_image_data_segment_temp);

    last_timestamp_ = mono_image_timestamp;

    return true;
}

void DataBase::print() {
    unique_lock<mutex> lock(m_data_mutex);

    LOG(INFO) << "imu_buffer_ size : " << imu_buffer_.size();
    LOG(INFO) << "wheel_buffer_ size : " << wheel_buffer_.size();
    LOG(INFO) << "gnss_buffer_ size : " << gnss_buffer_.size();
    LOG(INFO) << "mono_image_buffer_ size : " << mono_image_buffer_.size();
}

}  // namespace SensorFusion