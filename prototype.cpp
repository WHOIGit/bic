#include <iostream>
#include <string>
#include <fstream>
#include <ios>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <boost/format.hpp>
#include <boost/asio.hpp>
#include <boost/tokenizer.hpp>

#include "learn_correct.hpp"
#include "prototype.hpp"
#include "demosaic.hpp"
#include "illumination.hpp"
#include "interpolation.hpp"

using namespace std;
using namespace cv;

#define N_THREADS 12
#define PATH_FILE "aprs.csv"

void prototype::test_effective_resolution() {
  using cv::Mat;
  using std::cerr;
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
    cerr << k << "x (" << w << ", " << h << "): " << maxdiff << "m" << endl;
    stringstream outpaths;
    string outpath;
    outpaths << "blaz_" << k << ".png";
    outpath = outpaths.str();
    imwrite(outpath,Du*100);
  }
}

void in_flat_task(illum::Lightfield* frameAverage, boost::mutex* mutex, string inpath) {
  cerr << "POPPED " << inpath << endl;
  cv::Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH); // read input image
  if(cfa_LR.empty()) {
    return;
  }
  // convert to grayscale
  // FIXME hardcoded bayer pattern
  cv::Mat bgr_LR = demosaic(cfa_LR, "rggb") / 255; // lose the 16-bit bit depth
  cv::Mat y_LR(bgr_LR.size().height, bgr_LR.size().width, CV_32F);
  cv::cvtColor(bgr_LR, y_LR, CV_BGR2GRAY);
  if(y_LR.empty())
    throw std::runtime_error("grayscale conversion produced empty image");
  { // protect frame average
    boost::lock_guard<boost::mutex> lock(*mutex);
    frameAverage->addImage(y_LR);
  }
  cerr << "ADDED " << inpath << endl;
}

void out_flat_task(illum::Lightfield* frameAverage, boost::mutex* mutex, string inpath) {
  cerr << "POPPED " << inpath << endl;
  cv::Mat bgr_LR = imread(inpath);
  if(bgr_LR.empty()) {
    return;
  }
  // convert to grayscale
  cv::Mat y_LR(bgr_LR.size().height, bgr_LR.size().width, CV_32F);
  cv::cvtColor(bgr_LR, y_LR, CV_BGR2GRAY);
  if(y_LR.empty())
    throw std::runtime_error("grayscale conversion produced empty image");
  { // protect frame average
    boost::lock_guard<boost::mutex> lock(*mutex);
    frameAverage->addImage(y_LR);
  }
  cerr << "ADDED " << inpath << endl;
}

void prototype::test_flatness() {
  boost::mutex correctMutex;
  boost::mutex rawMutex;
  illum::Lightfield correctAverage;
  illum::Lightfield rawAverage;
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
  // post jobs
  ifstream inpaths(PATH_FILE); // FIXME hardcoded CSV file pathname
  string line;
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  vector<string> fields;
  while(getline(inpaths,line)) { // read pathames from a file
    Tokenizer tok(line);
    fields.assign(tok.begin(),tok.end());
    string inpath = fields.at(0);
    string outpath = fields.at(1) + ".png";
    if(access(outpath.c_str(),F_OK) != -1) {
      io_service.post(boost::bind(in_flat_task, &rawAverage, &rawMutex, inpath));
      cerr << "PUSHED " << inpath << endl;
      io_service.post(boost::bind(out_flat_task, &correctAverage, &correctMutex, outpath));
      cerr << "PUSHED " << outpath << endl;
    }
  }
  // destroy the work object to indicate that there are no more jobs
  work.reset();
  // now run all pending jobs to completion
  workers.join_all();

  std::cerr << "WRITING raw average" << std::endl;
  cv::Mat avg = rawAverage.getAverage();
  imwrite("avg_raw.tiff",avg);

  std::cerr << "WRITING correct average" << std::endl;
  avg = correctAverage.getAverage();
  imwrite("avg_correct.tiff",avg);
}
