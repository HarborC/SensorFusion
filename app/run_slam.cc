/*
 * @Author: Jiagang Chen
 * @Date: 2021-11-03 17:11:52
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-04 08:38:25
 * @Description: ...
 * @Reference: ...
 */
#include "common_header.h"
#include "fusion_system.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    auto config = YAML::LoadFile(argv[1]);

    // DatasetIoFactory DIF;
    auto interface =
        DatasetIO::DatasetIoFactory::getDatasetIo(std::string("euroc"));

    std::string datasetbase_path = config["datasetbase_path"].as<std::string>();
    interface->read(datasetbase_path);

    std::string save_path = datasetbase_path + std::string("/result");

    // FusionSystem Interface
    std::unique_ptr<SensorFusion::FusionSystem> fusion_system_interface(
        new SensorFusion::FusionSystem(std::string(argv[1])));

    // std::shared_ptr<std::thread> fusion_system_run_thread(new std::thread(
    //     &SensorFusion::FusionSystem::run, fusion_system_interface.get()));

    // Main loop
    int index_data = 0;
    double last_timestamp = -1;
    std::vector<std::vector<DatasetIO::Measurement::Ptr>> next_data;
    while (DatasetIO::DatasetIoFactory::getNextData(interface, next_data)) {
        // LOG(INFO) << "index data : " << index_data++ << " " <<
        // next_data.size();
        for (auto data_term : next_data) {
            if (data_term[0]->type == DatasetIO::MeasureType::kStereoImage) {
                std::dynamic_pointer_cast<DatasetIO::StereoImageData>(
                    data_term[0])
                    ->show(last_timestamp);
                last_timestamp =
                    std::dynamic_pointer_cast<DatasetIO::StereoImageData>(
                        data_term[0])
                        ->timestamp;
            }
        }
        // fusion_system_interface->feed_data(next_data);
    }

    return 0;
}
