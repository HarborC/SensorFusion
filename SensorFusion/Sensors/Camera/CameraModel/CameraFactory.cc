#include "CameraFactory.h"

#include <boost/algorithm/string.hpp>

#include "CataCamera.h"
#include "EquidistantCamera.h"
#include "LadybugCamera.h"
#include "PinholeCamera.h"
#include "PinholeFullCamera.h"
#include "ScaramuzzaCamera.h"

#include "ceres/ceres.h"

namespace camodocal {

std::shared_ptr<CameraFactory> CameraFactory::m_instance;

CameraFactory::CameraFactory() {}

std::shared_ptr<CameraFactory> CameraFactory::instance(void) {
    if (m_instance.get() == 0) {
        m_instance.reset(new CameraFactory);
    }

    return m_instance;
}

CameraPtr CameraFactory::generateCamera(Camera::ModelType modelType,
                                        const std::string& cameraName,
                                        cv::Size imageSize) const {
    switch (modelType) {
    case Camera::KANNALA_BRANDT: {
        EquidistantCameraPtr camera(new EquidistantCamera);

        EquidistantCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        camera->setParameters(params);
        return camera;
    }
    case Camera::PINHOLE: {
        PinholeCameraPtr camera(new PinholeCamera);

        PinholeCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        camera->setParameters(params);
        return camera;
    }
    case Camera::PINHOLE_FULL: {
        PinholeFullCameraPtr camera(new PinholeFullCamera);

        PinholeFullCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        camera->setParameters(params);
        return camera;
    }
    case Camera::SCARAMUZZA: {
        OCAMCameraPtr camera(new OCAMCamera);

        OCAMCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        camera->setParameters(params);
        return camera;
    }
    case Camera::LADYBUG: {
        LadybugCameraPtr camera(new LadybugCamera);

        LadybugCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        camera->setParameters(params);
        return camera;
    }
    case Camera::MEI:
    default: {
        CataCameraPtr camera(new CataCamera);

        CataCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        camera->setParameters(params);
        return camera;
    }
    }
}

CameraPtr CameraFactory::generateCameraFromYamlFile(
    const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        return CameraPtr();
    }

    Camera::ModelType modelType = Camera::MEI;
    if (!fs["model_type"].isNone()) {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (boost::iequals(sModelType, "kannala_brandt")) {
            modelType = Camera::KANNALA_BRANDT;
        } else if (boost::iequals(sModelType, "mei")) {
            modelType = Camera::MEI;
        } else if (boost::iequals(sModelType, "scaramuzza")) {
            modelType = Camera::SCARAMUZZA;
        } else if (boost::iequals(sModelType, "pinhole")) {
            modelType = Camera::PINHOLE;
        } else if (boost::iequals(sModelType, "PINHOLE_FULL")) {
            modelType = Camera::PINHOLE_FULL;
        } else if (boost::iequals(sModelType, "ladybug")) {
            modelType = Camera::LADYBUG;
        } else {
            std::cerr << "# ERROR: Unknown camera model: " << sModelType
                      << std::endl;
            return CameraPtr();
        }
    }

    switch (modelType) {
    case Camera::KANNALA_BRANDT: {
        EquidistantCameraPtr camera(new EquidistantCamera);

        EquidistantCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(filename);
        camera->setParameters(params);
        return camera;
    }
    case Camera::PINHOLE: {
        PinholeCameraPtr camera(new PinholeCamera);

        PinholeCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(filename);
        camera->setParameters(params);
        return camera;
    }
    case Camera::PINHOLE_FULL: {
        PinholeFullCameraPtr camera(new PinholeFullCamera);

        PinholeFullCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(filename);
        camera->setParameters(params);
        return camera;
    }
    case Camera::SCARAMUZZA: {
        OCAMCameraPtr camera(new OCAMCamera);

        OCAMCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(filename);
        camera->setParameters(params);
        return camera;
    }
    case Camera::LADYBUG: {
        LadybugCameraPtr camera(new LadybugCamera);

        LadybugCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(filename);
        camera->setParameters(params);
        return camera;
    }
    case Camera::MEI:
    default: {
        CataCameraPtr camera(new CataCamera);

        CataCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(filename);
        camera->setParameters(params);
        return camera;
    }
    }

    return CameraPtr();
}
}  // namespace camodocal
