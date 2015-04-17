//===============================================================================//
//                                                                               //
// This funtion builds a disparity map from a pair of rectified stereo images    //
// and then computes a point cloud in mm coordinates adjusted for water using    //
// the refraction index of salt water (1.33)                                     //
//                                                                               //
// Input:   image             Input image (rectified pair)                       //
//          Q                 Pixel to world coordinates transformation matrix   //
//          displayImage      A boolean option to display images as processed    //
//          pauseForKeystroke A boolean option to pause if image is displayed    //
// Output:  returned value    A 3 channel (x,y,z) point cloud matrix in mm       //
//                                                                               //
// Author:                    Peter Honig, phonig@whoi.edu, March 25 2015        //
//                                                                               //
//===============================================================================//

#include "Reconstruct3dImage.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;


PointCloud Reconstruct3dImage(Mat image, CameraMatrix cameraMatrix, bool displayImage, bool pauseForKeystroke)
{
	PointCloud pointCloud;
	Mat imageLeft, imageRight, disparity;
	Mat maskValid8U, maskValid, maskInvalid, maskOffset;
	Mat pointMatrix = Mat(1, 1, CV_32FC1), pointMatrix3D;
	double minVal, maxVal;

	double minMeanDisparity = 207.;			// lower limit of mean disparity (725500 / 207 disparity = 3500mm distance of camera)
	float waterRefractionIndex = 1.33f;		// 1.33 for salt water
	int trim = 25;							// amount to trim off each edge to because of image rotation

	// trim off pixels for black edges due to correction of lens distortion
	image = image(Rect(trim, trim, image.cols-(2*trim), image.rows-(2*trim))).clone();

	// if needed, convert to 8 bits per channel and scale intensity range to 0-255
	if (image.type() != CV_8UC3)
	{
		minMaxLoc(image, &minVal, &maxVal);
		image.convertTo(image, CV_8UC3, 255./(maxVal-minVal), -minVal*255./(maxVal-minVal));
	}

	// crop the image to remove right stereo half
	imageLeft = image(Rect(0, 0, image.cols/2, image.rows)).clone();
	imageRight = image(Rect(image.cols/2, 0, image.cols/2, image.rows)).clone();

	// call the StereoSGBM constructor (stereo correspondence Semi-Global Block Matching algorithm)
	// this is slow but very accurate and vastly superior to StereoBM (Block Matching algorithm)
	int minDisp = 0;			// 0 always
	int nDisparities = 16*25;	// 16*25 works best
	int SADWindowSize = 5;		// 1, 3, or 5 (5 works best by far)
	int p1 =  8 * 6 * SADWindowSize * SADWindowSize;
	int p2 = 32 * 6 * SADWindowSize * SADWindowSize;
	int disp12MaxDiff = 0;		// 0 always
	int uniquenessRatio = 1;	// 1 to 10
	int speckleWindowSize = 100;
	StereoSGBM sgbm(minDisp, nDisparities, SADWindowSize, p1, p2, disp12MaxDiff, 0, uniquenessRatio, speckleWindowSize , 2, true);

	// calculate the disparity (returned values are the pixel disparities multiplied by 16)
	sgbm(imageLeft, imageRight, disparity);

	// convert disparity to floating point pixel values and divide by 16 then add back the border that was previously trimmed
	disparity.convertTo(disparity, CV_32FC1, 1./16.);
	copyMakeBorder(disparity, disparity, trim, trim, trim, 0, BORDER_CONSTANT, Scalar(-1.f));

	// generate masks for all valid and invalid pixels using a min disparity at 3.5 meters (~407 disparity)
	threshold(disparity, maskValid, minMeanDisparity-(double)trim, 0.0, THRESH_TOZERO);				// disparity values at valid pixels
	threshold(disparity, maskInvalid, minMeanDisparity-(double)trim, -1.0, THRESH_BINARY_INV);		// value of -1 at invalid pixels
	threshold(disparity, maskOffset, minMeanDisparity-(double)trim, (double)trim, THRESH_BINARY);	// compensation for trim at valid pixels

	// sum all the masks to get disparity values + trim compensation at valid pixels and -1 at invalid pixels
	disparity = maskValid + maskOffset + maskInvalid;

	// FIRST ITERATION: compute mean value of all valid disparity pixels to determine the non-overlapping region of the left image
	maskValid.convertTo(maskValid8U, CV_8UC1);
	Scalar meanDisparityScalar = mean(disparity, maskValid8U);	// mask must be type CV_8UC1
	float meanDisparity = (float)meanDisparityScalar.val[0];

	// use mean disparty and edge trim to trim away regions where there is no disparity data,
	// then mask out the upper left of the remaining image with a triangle where lens distortion creates inaccuracies
	disparity = disparity(Rect((int)meanDisparity+trim, trim, disparity.cols-(int)meanDisparity-trim, disparity.rows-(2*trim))).clone();
	triangle(disparity, Point(0, 0), Point(0, 250), Point(150, 0), Scalar(-1.0));
	pointCloud.trimLeft = (int)meanDisparity+trim;
	pointCloud.trimRight = 0;
	pointCloud.trimTop = trim;
	pointCloud.trimBottom = trim;

	// SECOND ITERATION: get new mask for valid pixels, find the minimum and maximum disparities and the refine mean disparity
	threshold(disparity, maskValid, 0.0, 1.0, THRESH_TOZERO);	// mask values at valid pixels are 1, otherwise 0
	maskValid.convertTo(maskValid8U, CV_8UC1);
	minMaxLoc(disparity, &minVal, &maxVal, 0, 0, maskValid8U);	// mask must be type CV_8UC1
	meanDisparityScalar = mean(disparity, maskValid8U);
	meanDisparity = (float)meanDisparityScalar.val[0];

	// compute MIN distance in world coordinates (from MAXimum disparity)
	pointMatrix.at<float>(0,0) = (float) maxVal;
	reprojectImageTo3D(pointMatrix, pointMatrix3D, cameraMatrix.Q);
	pointCloud.minDistance = pointMatrix3D.at<Vec3f>(0,0)[2] * waterRefractionIndex;	// correct for water density

	// compute MAX distance in world coordinates (from MINimum disparity)
	pointMatrix.at<float>(0,0) = (float) minVal;
	reprojectImageTo3D(pointMatrix, pointMatrix3D, cameraMatrix.Q);
	pointCloud.maxDistance = pointMatrix3D.at<Vec3f>(0,0)[2] * waterRefractionIndex;	// correct for water density

	// compute mean distance in world coordinates
	pointMatrix.at<float>(0,0) =  meanDisparity;
	reprojectImageTo3D(pointMatrix, pointMatrix3D, cameraMatrix.Q);
	pointCloud.meanDistance = pointMatrix3D.at<Vec3f>(0,0)[2] * waterRefractionIndex;	// correct for water density

	// generate 3D point cloud from disparity map and correct for water density (adjust Z value only)
	reprojectImageTo3D(disparity, pointCloud.data, cameraMatrix.Q);
	vector<Mat> pointCloudChannels;
	split(pointCloud.data, pointCloudChannels);
	pointCloudChannels[2] = pointCloudChannels[2] * waterRefractionIndex;
	merge(pointCloudChannels, pointCloud.data);

	// get x and y range of 3D world coordinates of cloud
	minMaxLoc(pointCloudChannels[0], &minVal, &maxVal, 0, 0, maskValid8U);
	pointCloud.minX3D = (float) minVal;
	pointCloud.maxX3D = (float) maxVal;
	minMaxLoc(pointCloudChannels[1], &minVal, &maxVal, 0, 0, maskValid8U);
	pointCloud.minY3D = (float) minVal;
	pointCloud.maxY3D = (float) maxVal;

	// display the disparity map
	if (displayImage)
	{
		// display source image
		imshow("Input", image);

		// trim off left region where there is no disparity as well as the region where data is poor due to lens distortion
		//Mat disparityTemp = disparity(Rect((int)meanDisparity+trim, trim, disparity.cols-(int)meanDisparity-trim, disparity.rows-(2*trim))).clone();
		Mat disparityTemp = disparity;

		// compute the max and min disparity values for image intensity scaling
		threshold(disparityTemp, maskValid, 0.0, 1.0, THRESH_TOZERO);	// mask values at valid pixels are 1, otherwise 0
		maskValid.convertTo(maskValid8U, CV_8UC1);
		minMaxLoc(disparityTemp, &minVal, &maxVal, 0, 0, maskValid8U);	// mask must be type CV_8UC1
		
		// display as a normalized grayscale image with values 0 to 255
		Mat imageDisparity8U;
		double scale = 255./(maxVal-minVal);
		disparityTemp.convertTo(imageDisparity8U, CV_8UC1, scale, -minVal*scale);
		imshow("Disparity", imageDisparity8U);
		if (pauseForKeystroke)
		{
			char c = (char)waitKey();
			if( c == 27 || c == 'q' || c == 'Q' )	// allow for ESC or "q" to quit
				exit (-1);
		}

		// display as pseudo color image, initialize Sat and Val channels of HSV color space to 255
		Mat hue = imageDisparity8U;
		Mat sat = Mat(hue.rows, hue.cols, CV_8UC1, Scalar::all(255));
		Mat val = Mat(hue.rows, hue.cols, CV_8UC1, Scalar::all(255));
		
		// convert hue to degrees (actually degrees/2) and limit color value to blue (70%)
		hue.convertTo(hue, CV_8UC1, 180./256. * 0.70);

		// use a mask to make non-disparity data look black (Val = 0)
		multiply(val, maskValid8U, val);

		// flip the color map so near is red and far is blue
		Mat hueTemp;
		hue.convertTo(hueTemp, CV_16SC1);
		hueTemp = -1. * hueTemp;
		hueTemp = (179. * 0.70) + hueTemp;
		hueTemp.convertTo(hue, CV_8UC1);

		// merge the channels
		Mat imagePseudoColor;
		vector<Mat> channels;
		channels.push_back(imageDisparity8U);
		channels.push_back(sat);
		channels.push_back(val);
		merge(channels, imagePseudoColor);

		// convert to BGR color space and display
		cvtColor(imagePseudoColor, imagePseudoColor, CV_HSV2BGR);
		imshow("Disparity", imagePseudoColor);
		if (pauseForKeystroke)
		{
			char c = (char)waitKey();
			if( c == 27 || c == 'q' || c == 'Q' )	// allow for ESC or "q" to quit
				exit (-1);
		}
	}

	return pointCloud;
}


void triangle(Mat matrix, Point p1, Point p2, Point p3, Scalar fillValue)
{
	Point pointList[1][3];
	pointList[0][0] = p1;
	pointList[0][1] = p2;
	pointList[0][2] = p3;
	const Point* ppt[1] = {pointList[0]};
	int npt[] = {3};
	fillPoly(matrix, ppt, npt, 1, fillValue);
}
