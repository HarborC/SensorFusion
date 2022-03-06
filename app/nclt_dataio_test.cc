#include "common_header.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    auto config = YAML::LoadFile(argv[1]);

    // DatasetIoFactory DIF;
    auto interface =
        DatasetIO::DatasetIoFactory::getDatasetIo(std::string("nclt"));

    std::string datasetbase_path = config["datasetbase_path"].as<std::string>();
    interface->read(datasetbase_path);

    std::string save_path = datasetbase_path + std::string("/result");

    // other params
    int image_idx = 0;
    std::string voc_path = config["voc_path"].as<std::string>();
    int image_skip = config["image_skip"].as<int>();

    std::string cam_config_path =
        config["camera_config_path"].as<std::string>();
    auto camera_config = YAML::LoadFile(cam_config_path);

    // Main loop
    int index_data = 0;
    std::vector<std::vector<DatasetIO::Measurement::Ptr>> next_data;
    while (DatasetIO::DatasetIoFactory::getNextData(interface, next_data)) {
        LOG(INFO) << "index data : " << index_data++;
        for (size_t i = 0; i < next_data.size(); i++) {
            if (next_data[i][0]->type == DatasetIO::MeasureType::kMonoImage) {
                if (!(image_idx++ % image_skip == 0))
                    continue;
                std::vector<cv::Mat> images(1);

                images[0] = std::dynamic_pointer_cast<DatasetIO::MonoImageData>(
                                next_data[i][0])
                                ->data.clone();
                // cv::imshow("ws", images[0]);
                // cv::waitKey(0);
            } else if (next_data[i][0]->type ==
                       DatasetIO::MeasureType::kAccel) {
                auto accel_data =
                    std::dynamic_pointer_cast<DatasetIO::AccelData>(
                        next_data[i][0]);
                // std::cout << accel_data->data.transpose() << std::endl;
                auto gyro_data = std::dynamic_pointer_cast<DatasetIO::GyroData>(
                    next_data[i][1]);
                // std::cout << gyro_data->data.transpose() << std::endl;
            } else if (next_data[i][0]->type ==
                       DatasetIO::MeasureType::kWheel) {
                auto wheel_data =
                    std::dynamic_pointer_cast<DatasetIO::WheelData>(
                        next_data[i][0]);
                // std::cout << wheel_data->data.transpose() << std::endl;
            } else if (next_data[i][0]->type == DatasetIO::MeasureType::kGnss) {
                auto gnss_data = std::dynamic_pointer_cast<DatasetIO::GnssData>(
                    next_data[i][0]);
                // std::cout << gnss_data->data.transpose() << std::endl;
            }
        }
    }

    return 0;
}
