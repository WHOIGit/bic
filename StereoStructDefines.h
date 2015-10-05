//===============================================================================//
//                                                                               //
// Header file defining point cloud structure and enumeration                    //
//                                                                               //
// Author:                    Peter Honig, phonig@whoi.edu, March 1 2015         //
//                                                                               //
//===============================================================================//

#ifndef StereoStructDefines_H_
#define StereoStructDefines_H_

#include <opencv2/core/core.hpp>

// definition of point cloud data output file formats
enum FileFormat {PC_BINARY, PC_TEXT, PC_MESH, PC_MESH_TEXTURE};

struct PointCloud
{
	cv::Mat data;
	float meanDistance;
	float minDistance;
	float maxDistance;
	int trimLeft;
	int trimRight;
	int trimTop;
	int trimBottom;
	float minX3D;
	float maxX3D;
	float minY3D;
	float maxY3D;
} ;

struct CameraMatrix
{
	cv::Mat M1;
	cv::Mat D1;
	cv::Mat M2;
	cv::Mat D2;
	cv::Mat R;
	cv::Mat T;
	cv::Mat R1;
	cv::Mat R2;
	cv::Mat P1;
	cv::Mat P2;
	cv::Mat Q;
} ;

#endif