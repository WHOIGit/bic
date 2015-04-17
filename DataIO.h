//===============================================================================//
//                                                                               //
// Header for DataIO.cpp                                                         //
//                                                                               //
// Author:                    Peter Honig, phonig@whoi.edu, March 1 2015         //
//                                                                               //
//===============================================================================//

#ifndef DataIO_H_
#define DataIO_H_

#include "StereoStructDefines.h"
#include <opencv2/core/core.hpp>

#include <string>

bool ReadCameraMatrices(std::string calibrationDataDirectory, CameraMatrix &cameraMatrix);
bool ReadIntrinsicMatrices(std::string calibrationDataDirectory, CameraMatrix &cameraMatrix);
bool ReadExtrinsicMatrices(std::string calibrationDataDirectory, CameraMatrix &cameraMatrix);
bool WritePointCloud(std::string filename, PointCloud pointCloud, cv::Mat image, FileFormat fileFormat=PC_BINARY);

#endif