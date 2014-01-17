#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "prototype.hpp"
#include "interpolation.hpp"
#include "illumination.hpp"
#include "demosaic.hpp"

using namespace std;

void doit(cv::Mat y_LR_in) {
  using namespace cv;
  Mat y_LR;
  y_LR_in.convertTo(y_LR, CV_32F);
  // metrics
  int h = y_LR.size().height;
  int w = y_LR.size().width;
  int h2 = h/2; // half the height (center of image)
  int w2 = w/2; // half the width (split between image pair)
  int w4 = w/4; // 1/4 the width (center of left image)
  int w34 = w2 + w4; // 3/4 the width (center of right image)
  int ts = 64; // template size
  int ts2 = ts / 2;
  // now select a random template location in the left image
  uint64 seed = time(NULL);
  RNG rng(seed);
  Mat all(ts, w2-ts+1, CV_32F);
  Mat out(ts, w2-ts+1, CV_32F);
  for(int i = 0; i < 100; i++) {
    int x = rng.uniform(0,w2-ts*2);
    int y = rng.uniform(0,h-ts*2+1);
    // match the template against the right image
    Mat right(y_LR,Rect(w2,y,w2,ts*2-1));
    Mat templ(y_LR,Rect(x,y,ts,ts));
    matchTemplate(right, templ, out, CV_TM_CCOEFF);
    assert(all.size()==out.size());
    assert(all.type()==out.type());
    // now shift out by x
    Mat outR(out,Rect(x,0,w2-ts+1-x,ts));
    Mat outL(out,Rect(0,0,x,ts));
    Mat allL(all,Rect(0,0,w2-ts+1-x,ts));
    Mat allR(all,Rect(w2-ts+1-x,0,x,ts));
    allL += outR;
    allR += outL;
  }
  normalize(all,all,0,255,NORM_MINMAX);
  Mat out8u;
  all.convertTo(out8u,CV_8U);
  imwrite("match.tiff",out8u);
}

cv::Mat get_green(cv::Mat cfa_LR) {
  cv::Mat green;
  cfa_channel(cfa_LR, green, 1, 0);
  imwrite("green.tiff",green);
  return green;
}

int main(int argc, char **argv) {
  using namespace cv;
  Mat y_LR = imread(argv[1], CV_LOAD_IMAGE_ANYDEPTH);
  cout << y_LR.size().width << "," << y_LR.size().height << endl;
  Mat green = get_green(y_LR);
  cout << green.size().width << "," << green.size().height << endl;
  doit(green);
  /*
  if(string(argv[1])=="learn") {
    learn_prototype();
  } else {
    correct_prototype();
  }
  */
}
