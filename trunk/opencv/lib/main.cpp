#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <boost/tokenizer.hpp>
#include "prototype.hpp"
#include "interpolation.hpp"
#include "illumination.hpp"
#include "demosaic.hpp"

using namespace std;

int doit(cv::Mat y_LR_in) {
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
  /*
  Mat all(ts, w2-ts+1, CV_32F);
  Mat out(ts, w2-ts+1, CV_32F);
  for(int i = 0; i < 5; i++) {
    int x = rng.uniform(ts*2,w2-ts*4);
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
  Point minLoc, maxLoc;
  double minVal, maxVal;
  minMaxLoc(all,&minVal,&maxVal,&minLoc,&maxLoc);
  int max_x = maxLoc.x - ts;
  Mat out8u;
  all.convertTo(out8u,CV_8U);
  imwrite("match.tiff",out8u);
  return max_x;
  */
  int x = rng.uniform(ts*2,w2-ts*4);
  int y = rng.uniform(0,h-ts*2+1);
  Mat templ(y_LR,Rect(x,y,ts,ts));
  Mat right(y_LR,Rect(w2,0,w2,h-ts));
  Mat out;
  matchTemplate(right, templ, out, CV_TM_CCOEFF_NORMED);
  normalize(out,out,0,1,NORM_MINMAX);
  imwrite("match.tiff",out*255);
  Point minLoc, maxLoc;
  double minVal, maxVal;
  minMaxLoc(out,&minVal,&maxVal,&minLoc,&maxLoc);
  int max_x = maxLoc.x;
  int max_y = maxLoc.y;
  int mx = w2+max_x;
  int my = max_y;
  rectangle(y_LR_in,Point(x,y),Point(x+ts,y+ts),0);
  rectangle(y_LR_in,Point(mx,my),Point(mx+ts,my+ts),0);
  imwrite("pic.tiff",y_LR_in);
}

cv::Mat get_green(cv::Mat cfa_LR) {
  cv::Mat green;
  cfa_channel(cfa_LR, green, 1, 0);
  imwrite("green.tiff",green);
  return green;
}

int main(int argc, char **argv) {
  using namespace cv;
  /*
  Mat y_LR = imread(argv[1], CV_LOAD_IMAGE_ANYDEPTH);
  Mat green = get_green(y_LR);
  int x = doit(green);
  cout << argv[1] << "," << x << endl;
  */
  ifstream inpaths(argv[1]);
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  vector<string> fields;
  string line;
  while(getline(inpaths,line)) { // read pathames from a file
    Tokenizer tok(line);
    fields.assign(tok.begin(),tok.end());
    string inpath = fields.front();
    int offset = (int)(atoi(fields.back().c_str()));
    Mat y_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
    Mat green = get_green(y_LR);
    imwrite("green.tiff",green);
    int x = doit(green);
    cout << inpath << "," << offset << "," << x << endl;
    break;
  }
  /*
  if(string(argv[1])=="learn") {
    learn_prototype();
  } else {
    correct_prototype();
  }
  */
}
