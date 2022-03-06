#include "common_header.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    auto config = YAML::LoadFile(argv[1]);

    // DatasetIoFactory DIF;
    auto interface =
        DatasetIO::DatasetIoFactory::getDatasetIo(std::string("panovideo"));

    LOG(INFO) << "test panovideo";

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
    DatasetIO::MonoImageData::Ptr pano_data;
    while ((pano_data = (interface->get_data())->get_next_mono_image_data())) {
        if (!(image_idx++ % image_skip == 0))
            continue;

        std::vector<cv::Mat> images(1);
        images[0] = pano_data->data.clone();
    }

    return 0;
}

// void setupPoseOptProblem(bool perturbPose, bool rollingShutter,
//                          bool noisyKeypoint, int cameraObservationModelId) {
//   // srand((unsigned int) time(0));
//   swift_vio::CameraObservationOptions coo;
//   coo.perturbPose = perturbPose;
//   coo.rollingShutter = rollingShutter;
//   coo.noisyKeypoint = noisyKeypoint;
//   coo.cameraObservationModelId = cameraObservationModelId;

//   swift_vio::CameraObservationJacobianTest jacTest(coo);

//   okvis::ImuParameters imuParameters;
//   double imuFreq = imuParameters.rate;
//   Eigen::Vector3d ginw(0, 0, -imuParameters.g);
//   std::shared_ptr<simul::CircularSinusoidalTrajectory> cameraMotion(
//       new simul::WavyCircle(imuFreq, ginw));
//   okvis::ImuMeasurementDeque imuMeasurements;

//   okvis::Time startEpoch(2.0);
//   okvis::Time endEpoch(5.0);
//   cameraMotion->getTrueInertialMeasurements(startEpoch - okvis::Duration(1),
//                                             endEpoch + okvis::Duration(1),
//                                             imuMeasurements);
//   jacTest.addNavStatesAndExtrinsic(cameraMotion, startEpoch, 0.3);

//   double tdAtCreation(0.0);  // camera time offset used in initializing the
//   state time. double initialCameraTimeOffset(0.0);  // camera time offset's
//   initial estimate. double cameraTimeOffset(0.0);  // true camera time
//   offset. jacTest.addImuAugmentedParameterBlocks(startEpoch);
//   jacTest.addImuInfo(imuMeasurements, imuParameters, tdAtCreation);

//   std::shared_ptr<swift_vio::DistortedPinholeCameraGeometry> cameraGeometry =
//       std::static_pointer_cast<swift_vio::DistortedPinholeCameraGeometry>(
//           swift_vio::DistortedPinholeCameraGeometry::createTestObject());

//   Eigen::VectorXd intrinsicParams;
//   cameraGeometry->getIntrinsics(intrinsicParams);
//   double tr = 0;
//   if (jacTest.coo_.rollingShutter) {
//     tr = 0.03;
//   }
//   cameraGeometry->setReadoutTime(tr);
//   cameraGeometry->setImageDelay(cameraTimeOffset);
//   jacTest.addCameraParameterBlocks(intrinsicParams, startEpoch, tr,
//   initialCameraTimeOffset);

//   // get some random points
//   const size_t numberTrials = 200;
//   std::vector<std::shared_ptr<swift_vio::PointLandmark>> visibleLandmarks;
//   Eigen::AlignedVector<Eigen::AlignedVector<Eigen::Vector2d>>
//   pointObservationList;
//   jacTest.createLandmarksAndObservations(cameraGeometry, &visibleLandmarks,
//   &pointObservationList, numberTrials);

//   std::cout << "created " << visibleLandmarks.size()
//             << " visible points and add respective reprojection error
//             terms... "
//             << std::endl;

//   for (size_t i = 0u; i < visibleLandmarks.size(); ++i) {
//     jacTest.addLandmark(visibleLandmarks[i]);
//     for (size_t j = 0; j < pointObservationList[i].size(); ++j) {
//       std::shared_ptr<okvis::ImuMeasurementDeque> imuMeasDequePtr(
//           new okvis::ImuMeasurementDeque(imuMeasurements));

//       if (coo.cameraObservationModelId ==
//           swift_vio::cameras::kRsReprojectionErrorId)
//       {
//         std::shared_ptr<::ceres::CostFunction> costFunctionPtr;
//         std::shared_ptr<okvis::ceres::ErrorInterface> errorInterface;

//         std::shared_ptr<okvis::ceres::RsReprojectionError<
//             swift_vio::DistortedPinholeCameraGeometry,
//             swift_vio::ProjectionOptFXY_CXY, swift_vio::Extrinsic_p_BC_q_BC>>
//             localCostFunctionPtr(
//                 new okvis::ceres::RsReprojectionError<
//                     swift_vio::DistortedPinholeCameraGeometry,
//                     swift_vio::ProjectionOptFXY_CXY,
//                     swift_vio::Extrinsic_p_BC_q_BC>( cameraGeometry,
//                     pointObservationList[i][j], swift_vio::kCovariance,
//                     imuMeasDequePtr,
//                     std::shared_ptr<const Eigen::Matrix<double, 6, 1>>(),
//                     jacTest.stateEpoch(j), tdAtCreation, imuParameters.g));
//         costFunctionPtr =
//         std::static_pointer_cast<::ceres::CostFunction>(localCostFunctionPtr);
//         errorInterface =
//         std::static_pointer_cast<okvis::ceres::ErrorInterface>(localCostFunctionPtr);
//         jacTest.addResidual(costFunctionPtr, j, i);

//         std::shared_ptr<swift_vio::PointSharedData> pointDataPtr;
//         if (i % 20 == 0 && j == 2)
//         {
//           jacTest.verifyJacobians(errorInterface, j, i, pointDataPtr,
//                                   cameraGeometry,
//                                   pointObservationList[i][j]);
//         }
//       }
//       else if (coo.cameraObservationModelId ==
//                swift_vio::cameras::kRSCameraReprojectionErrorId)
//       {
//         std::shared_ptr<::ceres::CostFunction> costFunctionPtr;
//         std::shared_ptr<okvis::ceres::ErrorInterface> errorInterface;

//         std::shared_ptr<okvis::ceres::RSCameraReprojectionError<
//             swift_vio::DistortedPinholeCameraGeometry>>
//             localCostFunctionPtr(
//                 new okvis::ceres::RSCameraReprojectionError<
//                     swift_vio::DistortedPinholeCameraGeometry>(
//                     pointObservationList[i][j], swift_vio::kCovariance,
//                     cameraGeometry, imuMeasDequePtr, imuParameters,
//                     jacTest.stateEpoch(j), jacTest.stateEpoch(j) -
//                     tdAtCreation));
//         costFunctionPtr =
//         std::static_pointer_cast<::ceres::CostFunction>(localCostFunctionPtr);
//         errorInterface =
//         std::static_pointer_cast<okvis::ceres::ErrorInterface>(localCostFunctionPtr);
//         jacTest.addResidual(costFunctionPtr, j, i);

//         std::shared_ptr<swift_vio::PointSharedData> pointDataPtr;
//         if (i % 20 == 0 && j == 2)
//         {
//           jacTest.verifyJacobians(errorInterface, j, i, pointDataPtr,
//                                   cameraGeometry,
//                                   pointObservationList[i][j]);
//         }
//       }
//     }
//   }
//   std::cout << "Successfully constructed ceres solver pose optimization
//   problem." << std::endl;

//   jacTest.solveAndCheck();
// }
