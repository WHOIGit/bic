#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <boost/tokenizer.hpp>
#include <boost/regex.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/median.hpp>

#include "stereo.hpp"
#include "demosaic.hpp"

#define N_THREADS 12 // FIXME hardcoded thread count

int stereo::align(cv::Mat y_LR_in, int template_size) {
  using namespace std;
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
  int ts = template_size; // template size
  int ts2 = ts / 2;
  // generate a seed from image data so as:
  // - to make results reproducible per-image
  // - to reduce aliasing in cases where frames share non-moving objects
  double seedPixel = y_LR.at<double>(h2,w34); // center of right frame
  // now convert to raw bytes
  uint64 seedValue = uint64(seedPixel);
  RNG rng(seedValue); // always use same seed
  // take at least SAMPLE_SIZE samples
  int SAMPLE_SIZE=5, n=0;
  // try at least GIVE_UP times to get samples
  int GIVE_UP=100, g=0;
  using namespace boost::accumulators;
  accumulator_set<int, stats<tag::median > > samples;
  while(n < SAMPLE_SIZE && g < GIVE_UP) {
    g++; // every attempt counts towards giving up
    int x = rng.uniform(ts*2,w2-ts*4);
    int y = rng.uniform(0,h-ts*2+1);
    Mat templ(y_LR,Rect(x,y,ts,ts));
    Mat right(y_LR,Rect(w2,0,w2,h-ts));
    Mat out;
    matchTemplate(right, templ, out, CV_TM_CCOEFF_NORMED);
    normalize(out,out,0,1,NORM_MINMAX);
    Point minLoc, maxLoc;
    double minVal, maxVal;
    minMaxLoc(out,&minVal,&maxVal,&minLoc,&maxLoc);
    int max_x = maxLoc.x;
    int max_y = maxLoc.y;
    int mx = w2+max_x;
    int my = max_y;
    int xoff = x-(mx-w2);
    int ydiff = abs(max_y-y);
    if(xoff < 0 || ydiff > ts2) {
      continue;
    }
    // we have a sample, this counts towards sample size
    samples(x-(mx-w2));
    n++;
  }
  return median(samples);
}

cv::Mat get_green(cv::Mat cfa_LR) {
  cv::Mat green;
  cfa_channel(cfa_LR, green, 1, 0);
  return green;
}

void convert_task(std::string line) {
  using namespace cv;
  using namespace std;
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  vector<string> fields;
  Tokenizer tok(line);
  fields.assign(tok.begin(),tok.end());
  string inpath = fields.front();
  int offset = (int)(atoi(fields.back().c_str()));
  Mat y_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
  Mat green = get_green(y_LR);
  boost::regex re(".*/(.*)\\.tif");
  string fname = regex_replace(inpath,re,"\\1");
  green /= 255;
  Mat green_8u;
  green.convertTo(green_8u,CV_8U);
  boost::regex re2("FNAME"); // FIXME use boost::format
  string fgreen = regex_replace(string("xoff_testset/FNAME.png"),re2,fname);
  imwrite(fgreen,green_8u);
  cout << fname << ".png," << offset / 2 << endl;
}

void align_task(std::string line) {
  using namespace cv;
  using namespace std;
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  vector<string> fields;
  Tokenizer tok(line);
  fields.assign(tok.begin(),tok.end());
  string fname = fields.front();
  int offset = (int)(atoi(fields.back().c_str()));
  string inpath = "xoff_testset/";
  inpath += fname;
  Mat y_LR = imread(inpath);
  int x = stereo::align(y_LR);
  cout << fname << "," << offset << "," << x << endl;
}

void stereo::xoff_test(int argc, char **argv) {
  using namespace std;
  cout << "name,python,cpp" << endl;
  boost::asio::io_service io_service;
  ifstream inpaths(argv[1]);
  string line;
  while(getline(inpaths,line)) { // read pathames from a file
    io_service.post(boost::bind(align_task, line));
  }
  boost::thread_group workers; // workers
  for(int i = 0; i < N_THREADS; ++i) { // FIXME hardcoded thread count
    workers.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
  }
  while(getline(inpaths,line)) { // read pathames from a file
    io_service.post(boost::bind(align_task, line));
  }
  io_service.run();
}
