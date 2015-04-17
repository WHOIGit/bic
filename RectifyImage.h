//===============================================================================//
//                                                                               //
// Header for RectifyImage.cpp                                                   //
//                                                                               //
// Author:                    Peter Honig, phonig@whoi.edu, March 1 2015         //
//                                                                               //
//===============================================================================//

#ifndef RectifyImage_H_
#define RectifyImage_H_

#include "StereoStructDefines.h"
#include <opencv2/core/core.hpp>

cv::Mat RectifyImage(cv::Mat image, CameraMatrix cameraMatrix, bool displayImage=false, bool pauseForKeystroke=false);

#endif