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
#define PATH_FILE "aprs.csv"
#define MODEL_FILE "model.tiff"
#define BAYER_PATTERN "rggb"
#define OUT_DIR "out"
#define N_THREADS 12

// the learn task adds an image to a multilightfield model
void learn_task(MultiLightfield *model, string inpath, double alt, double pitch, double roll) {
  // get the input pathname
  cout << "POPPED " << inpath << " " << alt << "," << pitch << "," << roll << endl;
  // read the image (this can be done in parallel)
  Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
  cout << "Read " << inpath << endl;
  model->addImage(cfa_LR, alt, pitch, roll);
  cout << "Added " << inpath << endl;
}

// the correct task corrects images
void correct_task(MultiLightfield *model, string inpath, double alt, double pitch, double roll, string outpath) {
  cout << "POPPED " << inpath << " " << alt << "," << pitch << "," << roll << endl;
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
  boost::thread_group workers;
  // start up the work threads
  // use the work object to keep threads alive before jobs are posted
  // use auto_ptr so we can indicate that no more jobs will be posted
  auto_ptr<boost::asio::io_service::work> work(new boost::asio::io_service::work(io_service));
  // create the thread pool
  for(int i = 0; i < N_THREADS; i++) {
    workers.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
  }
  // now read input lines and post jobs
  ifstream inpaths(PATH_FILE);
  string line;
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  vector<string> fields;
  while(getline(inpaths,line)) { // read pathames from a file
    Tokenizer tok(line);
    fields.assign(tok.begin(),tok.end());
    string inpath = fields.at(0);
    double alt = atof(fields.at(1).c_str()); // altitude in m
    double pitch_deg = atof(fields.at(2).c_str());
    double roll_deg = atof(fields.at(3).c_str());
    double pitch = M_PI * pitch_deg / 180.0;
    double roll = M_PI * roll_deg / 180.0;
    io_service.post(boost::bind(learn_task, &model, inpath, alt, pitch, roll));
    cout << "PUSHED " << inpath << endl;
  }
  // destroy the work object to indicate that there are no more jobs
  work.reset();
  // now run all pending jobs to completion
  workers.join_all();
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
  // use the work object to keep threads alive before jobs are posted
  // use auto_ptr so we can indicate that no more jobs will be posted
  auto_ptr<boost::asio::io_service::work> work(new boost::asio::io_service::work(io_service));
  // start a thread pool
  boost::thread_group workers;
  for(int i = 0; i < N_THREADS; i++) {
    workers.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
  }
  // post jobs
  ifstream inpaths(PATH_FILE);
  string line;
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  vector<string> fields;
  int count = 0;
  while(getline(inpaths,line)) { // read pathames from a file
    if(count % 5 == 0) {
      Tokenizer tok(line);
      fields.assign(tok.begin(),tok.end());
      string inpath = fields.at(0);
      double alt = atof(fields.at(1).c_str()); // altitude in m
      double pitch_deg = atof(fields.at(2).c_str());
      double roll_deg = atof(fields.at(3).c_str());
      double pitch = M_PI * pitch_deg / 180.0;
      double roll = M_PI * roll_deg / 180.0;
      stringstream outpaths;
      string outpath;
      outpaths << OUT_DIR << "/correct" << count << ".tiff";
      outpath = outpaths.str();
      io_service.post(boost::bind(correct_task, &model, inpath, alt, pitch, roll, outpath));
      cout << "PUSHED " << inpath << endl;
    }
    count++;
  }
  // destroy the work object to indicate that there are no more jobs
  work.reset();
  // now run all pending jobs to completion
  workers.join_all();
  // we're done
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

void prototype::test_effective_resolution() {
  using cv::Mat;
  using std::cout;
  using std::endl;

  // metrics: pixels
  int width_px = 1360;
  int height_px = 1024;
  // metrics: meters
  double pixel_sep = 0.0000065;
  double width = width_px * pixel_sep;
  double height = height_px * pixel_sep;
  double focal_length = 0.012;

  // FIXME choose alt, pitch, and roll randomly
  double alt = 1.3;
  double pitch = M_PI * -17 / 180.0;
  double roll = M_PI * 33 / 180.0;

  // compute full-resolution distance map

  Mat D = Mat::zeros(height_px, width_px, CV_32F);
  interp::distance_map(D, alt, pitch, roll, width, height, focal_length);

  // now compute at a series of reduced resolutions
  for(int e = 1; e < 6; e++) {
    // calculate downscaled resolution
    int k = pow(2,e); // 2^e
    int w = width_px / k;
    int h = height_px / k;
    // compute distance map at downscaled resolution
    Mat Dd = Mat::zeros(h, w, CV_32F);
    interp::distance_map(Dd, alt, pitch, roll, width, height, focal_length);
    // upscale using high-quality interpolation
    Mat Du;
    cv::resize(Dd, Du, D.size(), 0, 0, CV_INTER_CUBIC);
    Mat DuD = cv::abs(Du - D);
    double mindiff, maxdiff;
    cv::minMaxLoc(DuD, &mindiff, &maxdiff);
    cout << k << "x (" << w << ", " << h << "): " << maxdiff << "m" << endl;
    stringstream outpaths;
    string outpath;
    outpaths << "blaz_" << k << ".png";
    outpath = outpaths.str();
    imwrite(outpath,Du*100);
  }
}
