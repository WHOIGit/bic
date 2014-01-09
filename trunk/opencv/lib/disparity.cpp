#include <opencv2/opencv.hpp>

using cv::Mat;

void disparity(Mat left, Mat right, Mat disp) {
  CvStereoBMState* BMState;
  CvMat cvLeft = left;
  CvMat cvRight = right;
  CvMat cvDisp = disp;
  BMState = cvCreateStereoBMState(CV_STEREO_BM_BASIC,64);
  cvFindStereoCorrespondenceBM(&cvLeft,&cvRight,&cvDisp,BMState);
}
