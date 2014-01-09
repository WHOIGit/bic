#include <string>
#include <opencv2/opencv.hpp>
#include "disparity.hpp"
#include "demosaic.hpp"

using std::string;
using namespace cv;

int main(int argc, char **argv) {
  Mat y_LR, bgr_LR = imread(argv[1]);
  cvtColor(bgr_LR, y_LR, CV_BGR2GRAY);
  // now we can do stereo correspondence
  int w = y_LR.size().width / 2;
  int h = y_LR.size().height;
  Mat left(y_LR, Rect(0, 0, w, h));
  Mat right(y_LR, Rect(w, 0, w, h));
  Mat disp(h, w, CV_16S);
  disparity(left, right, disp);
  imwrite("disparity.tiff", disp);
}
