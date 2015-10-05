//===============================================================================//
//                                                                               //
// This program rectifies joined pairs of stereo images based on a list of files //
// where each line contains in input file followed by an output file that are    //
// separated by a comma, and them writes them to the specifed destination.       //
// Then rectified images are used to create a disparity map which in turn is     //
// used to create a 3D point cloud.                                              //
//                                                                               //
// Input:   image             Input image pair (joined left and right)           //
//          cameraMatrix      Camera calibration matrices                        //
//          doNotRectify      A boolean option to skip the rectification process //
//          displayRectifiedImage Boolean option to display images as processed  //
//          displayDisparityImage Boolean option to display images as processed  //
//          pauseForKeystroke A boolean option to pause if image is displayed    //
// Output:  imageRectified    A joined left-right pair of rectified images       //
//          pointCloud        A 3D world coordinate reconstruction in mm units   //
//          returned value    The mean altitude of camera in mm units            //
//                                                                               //
// Author:                  Peter Honig, phonig@whoi.edu, April 15 2015          //
//                                                                               //
//===============================================================================//

#include "StereoStructDefines.h"
#include "RectifyImage.h"
#include "Reconstruct3dImage.h"
#include "DataIO.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;


float AltitudeFromStereo(Mat image, CameraMatrix cameraMatrix, Mat &imageRectified, PointCloud &pointCloud,
	bool doNotRectify, bool displayRectifiedImage, bool displayDisparityImage, bool pauseForKeystroke)
{
	// rectify the image pair
	if (doNotRectify)
		imageRectified = image;
	else
		imageRectified = RectifyImage(image, cameraMatrix, displayRectifiedImage, pauseForKeystroke);

	// generate a point cloud
	pointCloud = Reconstruct3dImage(imageRectified, cameraMatrix, displayDisparityImage, pauseForKeystroke);

	// return the altitude
	return pointCloud.meanDistance;
}
