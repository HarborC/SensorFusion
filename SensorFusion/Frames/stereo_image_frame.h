/*
 * @Author: Jiagang Chen
 * @Date: 2021-11-04 12:38:46
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-04 12:40:07
 * @Description: ...
 * @Reference: ...
 */
/*
 * @Author: Jiagang Chen
 * @Date: 2021-11-04 12:36:54
 * @LastEditors: Jiagang Chen
 * @LastEditTime: 2021-11-04 12:37:59
 * @Description: ...
 * @Reference: ...
 */
#ifndef _SENSOR_FUSION_STEREO_IMAGE_FRAME_H_
#define _SENSOR_FUSION_STEREO_IMAGE_FRAME_H_

#include "../common_header.h"
#include "frame.h"

namespace SensorFusion {

class StereoImageFrame : public Frame{
public:
    StereoImageFrame();
    ~StereoImageFrame() {}

public:
    
};

}  // namespace SensorFusion

#endif