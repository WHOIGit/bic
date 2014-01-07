#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

Mat demosaic_hqlinear(Mat cfa, string cfaPattern="rggb");
