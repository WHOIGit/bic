//===============================================================================//
//                                                                               //
// Header for Reconstruct3dImage.cpp                                             //
//                                                                               //
// Author:                    Peter Honig, phonig@whoi.edu, March 25 2015        //
//                                                                               //
//===============================================================================//

#ifndef Reconstruct3dImage_H_
#define Reconstruct3dImage_H_

#include "StereoStructDefines.h"
#include <opencv2/core/core.hpp>

#include <vector>

PointCloud Reconstruct3dImage(cv::Mat image, CameraMatrix cameraMatrix, bool displayImage=false, bool pauseForKeystroke=false);
void triangle(cv::Mat matrix, cv::Point p1, cv::Point p2, cv::Point p3, cv::Scalar fillValue);

#endif