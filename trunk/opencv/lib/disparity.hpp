#pragma once
#include <opencv2/opencv.hpp>

cv::Mat get_L(cv::Mat LR);
cv::Mat get_R(cv::Mat LR);
void disparity(cv::Mat left, cv::Mat right, cv::Mat disp);
int xOffset(cv::Mat y_LR);
