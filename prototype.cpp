#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/tokenizer.hpp>

#include "prototype.hpp"
#include "demosaic.hpp"
#include "illumination.hpp"
#include "interpolation.hpp"

using namespace std;
using namespace cv;

using illum::MultiLightfield;

// this is a prototype application; code is not in reusable state yet

// hardcoded input and output parameters
#define PATH_FILE "alts.csv"
#define MODEL_FILE "model.tiff"
#define BAYER_PATTERN "rggb"
#define OUT_DIR "out"
#define N_THREADS 12

// the learn task adds an image to a multilightfield model
void learn_task(MultiLightfield *model, string inpath, double alt, double pitch, double roll) {
  static boost::mutex mutex; // shared lock for lightfield (is static sufficient?)
  // get the input pathname
  cout << "POPPED " << inpath << " " << alt << endl;
  // read the image (this can be done in parallel)
  Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
  cout << "Read " << inpath << endl;
  { // now lock the lightfield just long enough to add the image
    boost::lock_guard<boost::mutex> lock(mutex);
    model->addImage(cfa_LR, alt, pitch, roll);
  }
  cout << "Added " << inpath << endl;
}

// the correct task corrects images
void correct_task(MultiLightfield *model, string inpath, double alt, double pitch, double roll, string outpath) {
  cout << "POPPED " << inpath << " " << alt << endl;
  Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH); // read input image
  // get the average
  Mat average = Mat::zeros(cfa_LR.size(), CV_32F);
  model->getAverage(average, alt, pitch, roll);
  // now smooth the average
  int h = average.size().height;
  int w = average.size().width;
  Mat left = Mat(average,Rect(0,0,w/2,h));
  Mat right = Mat(average,Rect(w/2,0,w/2,h));
  cfa_smooth(left,left,31);
  cfa_smooth(right,right,31);
  cout << "SMOOTHED lightmap" << endl;
  illum::correct(cfa_LR, cfa_LR, average); // correct it
  cout << "Demosaicing " << inpath << endl;
  Mat rgb_LR = demosaic(cfa_LR,BAYER_PATTERN); // demosaic it
  cout << "Saving RGB to " << outpath << endl; // save as 16-bit
  Mat rgb_LR_16u;
  rgb_LR.convertTo(rgb_LR_16u, CV_16U);
  imwrite(outpath, rgb_LR_16u); // FIXME should we save as 8-bit?
}

// learn phase
void prototype::learn() {
  // construct an empty lightfield model
  illum::MultiLightfield model;
  // post all work
  boost::asio::io_service io_service;
  ifstream inpaths(PATH_FILE);
  string line;
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  vector<string> fields;
  while(getline(inpaths,line)) { // read pathames from a file
    Tokenizer tok(line);
    fields.assign(tok.begin(),tok.end());
    string inpath = fields.front();
    double alt = atof(fields.back().c_str()); // altitude in m
    double pitch = 0;
    double roll = 0;
    io_service.post(boost::bind(learn_task, &model, inpath, alt, pitch, roll));
    cout << "PUSHED " << inpath << endl;
  }
  boost::thread_group workers;
  // start up the work threads
  for(int i = 0; i < N_THREADS; i++) {
    workers.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
  }
  // now run all work to completion
  io_service.run();
  cout << "SUCCESS learn phase" << endl;

  model.save(OUT_DIR);
  cout << "SAVED model" << endl;
}

void prototype::correct() {
  // load model
  cout << "LOADING model..." << endl;
  illum::MultiLightfield model;
  model.load(OUT_DIR);
  cout << "LOADED model" << endl;

  // post all work
  boost::asio::io_service io_service;
  ifstream inpaths(PATH_FILE);
  string line;
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  vector<string> fields;
  int count = 0;
  while(getline(inpaths,line)) { // read pathames from a file
    if(count % 5 == 0) {
      Tokenizer tok(line);
      fields.assign(tok.begin(),tok.end());
      string inpath = fields.front();
      double alt = atof(fields.back().c_str());
      double pitch = 0;
      double roll = 0;
      stringstream outpaths;
      string outpath;
      outpaths << OUT_DIR << "/correct" << count << ".tiff";
      outpath = outpaths.str();
      io_service.post(boost::bind(correct_task, &model, inpath, alt, pitch, roll, outpath));
      cout << "PUSHED " << inpath << endl;
    }
    count++;
  }
  boost::thread_group workers;
  // start up the work threads
  for(int i = 0; i < N_THREADS; i++) {
    workers.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
  }
  // now run all work to completion
  io_service.run();
  cout << "SUCCESS correct phase" << endl;
}

// test altitude pitch roll code
void prototype::test_distance_map() {
  using cv::Mat;
  using std::cout;
  using std::endl;
  double alt = 0.7;
  double pitch = M_PI * 0 / 180.0;
  double roll = M_PI * 45 / 180.0;
  double pixel_sep = 0.0000065;
  double width = 1360 * pixel_sep;
  double height = 1024 * pixel_sep;
  double focal_length = 0.012;

  Mat D = Mat::zeros(1024/4, 1360/4, CV_32F);
  interp::distance_map(D, alt, pitch, roll, width, height, focal_length);

  double minD, maxD;
  cv::minMaxLoc(D, &minD, &maxD);

  double delta = 0.1;

  Mat W = Mat::zeros(D.size(), CV_32F);
  for(int i = 0; i < 12; i++) {
    interp::dist_weight(D, W, delta, i);
    if(cv::countNonZero(W) == 0) {
      cout << i << " is empty" << endl;
    } else {
      stringstream outpaths;
      string outpath;
      outpaths << "weight_" << i << ".png";
      imwrite(outpaths.str(), W * 255);
    }
  }

  // now generate some products to test that pitch/roll are oriented
  // correctly
  // compare D to jrock's algorithm
  int w = 1360;
  int h = 1024;
  cv::Mat Djr = interp::jrock_pitch_roll(alt, pitch, roll, w, h, w, h, focal_length, pixel_sep);
  double minDjr, maxDjr;
  cv::minMaxLoc(Djr, &minDjr, &maxDjr);

  imwrite("Djf.png",(D-0.5)*200);
  imwrite("Djr.png",(Djr-0.5)*200);
}
