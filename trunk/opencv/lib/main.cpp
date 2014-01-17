#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "prototype.hpp"
#include "interpolation.hpp"
#include "illumination.hpp"
#include "demosaic.hpp"

using namespace std;

void doit(string fname) {
  using namespace cv;
  Mat y_LR_in = imread(fname);
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
  // perform template matching
  Mat out = Mat::zeros(h,w2+ts+1,CV_32F);
  int oxen[3] = { 0, ts, 0-ts };
  for(int y = 0; y < h-(ts*2); y += ts+1) {
    for(int oxi = 0; oxi < 3; oxi++) {
      int ox = oxen[oxi];
      // select the center pixels of the left image
      Mat templ = Mat(y_LR, Rect((w4+ox)-ts2,y,ts,ts));
      // now match that template against the corresponding horizontal
      // strip of right image and accumulate into and output "strips"
      // image
      // image is w2 x ts*2. template is ts x ts
      // outstrip is w2+1 x ts+1
      Mat instrip = Mat(y_LR, Rect(w2,y,w2,ts*2));
      Mat outstrip = Mat::zeros(ts+1, w2-ts+1, CV_32F);
      matchTemplate(instrip, templ, outstrip, CV_TM_SQDIFF_NORMED);
      Mat roi = Mat(out, Rect(ts+ox,y,w2-ts+1,ts+1));
      roi += outstrip;
    }
    // now normalize this strip
    //Mat roi = Mat(out, Rect(ts,y,w2-ts,ts+1));
    //normalize(roi,roi,0,1,NORM_MINMAX);
  }
  Mat clipped(out, Rect(ts*2,0,w2-(ts*3),h));
  normalize(clipped,clipped,1,0,NORM_MINMAX);
  clipped *= 255;
  Mat out8u;
  clipped.convertTo(out8u, CV_8U);
  out /= 255;
  imwrite("strips.tiff",out8u);
  // sum all rows of out image
  Mat summed;
  reduce(out, summed, 0, CV_REDUCE_SUM, CV_32F);
  // now find the x location of maximum of this sum
  Point minLoc, maxLoc;
  double minVal, maxVal;
  minMaxLoc(summed, &minVal, &maxVal, &minLoc, &maxLoc);
  int max_x = (maxLoc.x - ts2)/2;
  int dx = w4 - max_x;
  cout << "found match at " << max_x << ", offset = " << dx << endl;
}

int main(int argc, char **argv) {
  doit(argv[1]);
  /*
  if(string(argv[1])=="learn") {
    learn_prototype();
  } else {
    correct_prototype();
  }
  */
}
