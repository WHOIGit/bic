#include <opencv2/opencv.hpp>

using cv::Mat;
using cv::Scalar;
using cv::Rect;

Mat get_L(Mat LR) {
  int w = LR.size().width / 2;
  int h = LR.size().height;
  return Mat(LR, Rect(0, 0, w, h));
}

Mat get_R(Mat LR) {
  int w = LR.size().width / 2;
  int h = LR.size().height;
  return Mat(LR, Rect(w, 0, w, h));
}

void disparity(Mat left, Mat right, Mat disp) {
  CvStereoBMState* BMState;
  CvMat cvLeft = left;
  CvMat cvRight = right;
  CvMat cvDisp = disp;
  BMState = cvCreateStereoBMState(CV_STEREO_BM_BASIC,64);
  // defaults. FIXME hand-tuned
  BMState->preFilterSize            = 5;
  BMState->preFilterCap             = 1;
  BMState->SADWindowSize            = 65;
  BMState->minDisparity             = -65;
  BMState->numberOfDisparities      = 128;
  BMState->textureThreshold         = 0;
  BMState->uniquenessRatio          = 0;
  BMState->speckleWindowSize        = 0;
  BMState->speckleRange             = 0;
  cvFindStereoCorrespondenceBM(&cvLeft,&cvRight,&cvDisp,BMState);
}

int xOffset(Mat y_LR) {
  int w = y_LR.size().width / 2;
  int h = y_LR.size().height;
  Mat left(y_LR, Rect(0, 0, w, h));
  Mat right(y_LR, Rect(w, 0, w, h));
  Mat disp(h, w, CV_16S);
  disparity(left, right, disp);
  // now compute the average disparity
  Mat mask = (disp > -200) & (disp < 2000); // but not where it's out of range
  // FIXME compute the median instead
  Scalar avg = cv::mean(disp, mask);
  return avg[0];
}
