//===============================================================================//
//                                                                               //
// These functions all perform some sort of file-based I/O                       //
//                                                                               //
// Author:                    Peter Honig, phonig@whoi.edu, March 1 2015         //
//                                                                               //
//===============================================================================//

#include "StereoStructDefines.h"
#include "DataIO.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

bool ReadCameraMatrices(string calibrationDataDirectory, CameraMatrix &cameraMatrix)
{
	// get the intrinsic and extrinsic matrices from previous calibration
	if (!ReadIntrinsicMatrices(calibrationDataDirectory, cameraMatrix) || !ReadExtrinsicMatrices(calibrationDataDirectory, cameraMatrix))
		return false;
	else
		return true;
}


bool ReadIntrinsicMatrices(string calibrationDataDirectory, CameraMatrix &cameraMatrix)
{
	// get the intrinsic matrices from previous calibration
	FileStorage fs(calibrationDataDirectory + "/intrinsics.yml", CV_STORAGE_READ);
	if (fs.isOpened())
	{
		fs["M1"] >> cameraMatrix.M1;
		fs["D1"] >> cameraMatrix.D1;
		fs["M2"] >> cameraMatrix.M2;
		fs["D2"] >> cameraMatrix.D2;
		fs.release();
		return true;
	}
	else
		return false;
}


bool ReadExtrinsicMatrices(string calibrationDataDirectory, CameraMatrix &cameraMatrix)
{
	// get the extrinsic matrices from previous calibration
	FileStorage fs(calibrationDataDirectory + "/extrinsics.yml", CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["R"] >> cameraMatrix.R;
		fs["T"] >> cameraMatrix.T;
		fs["R1"] >> cameraMatrix.R1;
		fs["R2"] >> cameraMatrix.R2;
		fs["P1"] >> cameraMatrix.P1;
		fs["P2"] >> cameraMatrix.P2;
		fs["Q"] >> cameraMatrix.Q;
		fs.release();
		return true;
	}
	else
		return false;
}


bool WritePointCloud(string filename, PointCloud pointCloud, Mat image, FileFormat fileFormat)
{
	if (fileFormat == PC_TEXT)
	{
		// open an ascii file for output
		ofstream fout;
		fout.open((filename+".txt").c_str());
		if(!fout.good() || fout.bad())
			return false;

		// write out header indicating number of rows and columns
		Mat data = pointCloud.data;
		fout << "Rows " << data.rows << endl;
		fout << "Columns " << data.cols << endl;
		fout << "Order_By_Row" << endl;
		fout << "Mean_Distance " << pointCloud.meanDistance << endl;
		fout << "Min_Distance " << pointCloud.minDistance << endl;
		fout << "Max_Distance " << pointCloud.maxDistance << endl;

		// write the integer values one at a time by rows
		for (int iRow=0; iRow<data.rows; iRow++)
		{
			for (int iCol=0; iCol<data.cols; iCol++)
			{
				if (data.at<Vec3f>(iRow,iCol)[2] <= 0.0f)
					fout << 0 << " " << 0 << " " << 0 << endl;
				else
					fout << data.at<Vec3f>(iRow,iCol)[0] << " " << data.at<Vec3f>(iRow,iCol)[1] << " " << data.at<Vec3f>(iRow,iCol)[2] << endl;	// x,y,z coordinates
			}
		}

		// close the file
		fout.close();
	}
	else if (fileFormat == PC_MESH || fileFormat == PC_MESH_TEXTURE)
	{
		// perspective corrected output with optional texture overlay image
		float xMax = pointCloud.maxX3D;
		float xMin = pointCloud.minX3D;
		float xRange = xMax - xMin;
		float yMax = pointCloud.maxY3D;
		float yMin = pointCloud.minY3D;
		float yRange = yMax - yMin;
		float perspective;
		
		// open an ascii file for output
		ofstream fout;
		fout.open((filename+".xyz").c_str());
		if(!fout.good() || fout.bad())
			return false;

		// write out header indicating number of rows and columns
		Mat data = pointCloud.data;
		fout << "AT3D_XYZ V01.00 Mesh A" << endl;
		fout << "Rows " << data.rows << endl;
		fout << "Columns " << data.cols << endl;
		fout << "Order By_Row" << endl;
		fout << "Data" << endl;

		// write the integer values one at a time by rows
		for (int iRow=0; iRow<data.rows; iRow++)
		{
			for (int iCol=0; iCol<data.cols; iCol++)
			{
				if (data.at<Vec3f>(iRow,iCol)[2] <= 0.0f)
				{
					// non-valid disparity x,y,z coordinates
					fout << xMin + (xRange*(float)iCol/(float)data.cols) << " ";
					fout << yMin + (yRange*(float)iRow/(float)data.rows) << " ";
					fout << pointCloud.maxDistance << endl;
				}
				else
				{
					// perspective-corrected x,y,z coordinates
					perspective = pointCloud.meanDistance / data.at<Vec3f>(iRow,iCol)[2];
					fout << data.at<Vec3f>(iRow,iCol)[0] * perspective << " ";
					fout << data.at<Vec3f>(iRow,iCol)[1] * perspective << " ";
					fout << data.at<Vec3f>(iRow,iCol)[2] << endl;
				}
			}
		}

		// close the file
		fout.close();

		// write texture file for mapping onto point cloud mesh
		if (fileFormat == PC_MESH_TEXTURE)
		{
			// trim and write the companion texture file
			image = image(Rect(pointCloud.trimLeft, pointCloud.trimTop,
				(image.cols/2)-pointCloud.trimLeft-pointCloud.trimRight,
				image.rows-pointCloud.trimTop-pointCloud.trimBottom));
			if (!imwrite(filename+"Texture.jpg", image))
				return false;
		}
	}
	else
	{
		float xMax = pointCloud.maxX3D;
		float xMin = pointCloud.minX3D;
		float xRange = xMax - xMin;
		float yMax = pointCloud.maxY3D;
		float yMin = pointCloud.minY3D;
		float yRange = yMax - yMin;

		// open a binary file for output
		ofstream fout;
		fout.open((filename+".dat").c_str(), ios::binary);
		if(!fout.good() || fout.bad())
			return false;

		// write out header indicating number of rows, columns, and camera-to-subject distances
		Mat data = pointCloud.data;
		float meanDistance = pointCloud.meanDistance;
		float minDistance = pointCloud.minDistance;
		float maxDistance = pointCloud.maxDistance;
		fout.write((char*)&data.rows, sizeof(data.rows));		// int
		fout.write((char*)&data.cols, sizeof(data.cols));		// int
		fout.write((char*)&meanDistance, sizeof(meanDistance));	// float
		fout.write((char*)&minDistance, sizeof(minDistance));	// float
		fout.write((char*)&maxDistance, sizeof(maxDistance));	// float

		// write the integer values one at a time by rows
		float x, y, z, zero = 0.0f;
		for (int iRow=0; iRow<data.rows; iRow++)
		{
			for (int iCol=0; iCol<data.cols; iCol++)
			{
				if (data.at<Vec3f>(iRow,iCol)[2] <= 0.0f)
				{
					// non-valid disparity x,y,z coordinates
					x = xMin + (xRange*(float)iCol/(float)data.cols);
					y = yMin + (yRange*(float)iRow/(float)data.rows);
					fout.write((char*)&x, sizeof(x));			// float
					fout.write((char*)&y, sizeof(y));			// float
					fout.write((char*)&zero, sizeof(zero));		// float
				}
				else
				{
					// x, y, z coordinates
					x = data.at<Vec3f>(iRow,iCol)[0];
					y = data.at<Vec3f>(iRow,iCol)[1];
					z = data.at<Vec3f>(iRow,iCol)[2];
					fout.write((char*)&x, sizeof(x));			// float
					fout.write((char*)&y, sizeof(y));			// float
					fout.write((char*)&z, sizeof(z));			// float
				}
			}
		}
		// close the file
		fout.close();
	}

	// done
	return true;
}

