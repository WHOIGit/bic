#pragma once
#include <opencv2/opencv.hpp>

cv::Mat demosaic_hqlinear(cv::Mat cfa, std::string cfaPattern="rggb");
