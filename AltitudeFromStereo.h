//===============================================================================//
//                                                                               //
// Header for AltitudeFromStereo.cpp                                             //
//                                                                               //
// Author:                    Peter Honig, phonig@whoi.edu, April 15 2015        //
//                                                                               //
//===============================================================================//

#ifndef AltitudeFromStereo_H_
#define AltitudeFromStereo_H_

#include "StereoStructDefines.h"
#include <opencv2/core/core.hpp>

float AltitudeFromStereo(cv::Mat image, CameraMatrix cameraMatrix, cv::Mat &imageRectified, PointCloud &pointCloud,
	bool doNotRectify, bool displayRectifiedImage, bool displayDisparityImage, bool pauseForKeystroke);

#endif