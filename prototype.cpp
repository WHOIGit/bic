#include <iostream>
#include <string>
#include <fstream>
#include <ios>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <boost/format.hpp>
#include <boost/asio.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "learn_correct.hpp"
#include "prototype.hpp"
#include "demosaic.hpp"
#include "illumination.hpp"
#include "interpolation.hpp"
#include "logging.hpp"

namespace fs = boost::filesystem;

using namespace std;
using namespace cv;

using jlog::log;

#define N_THREADS 12
#define PATH_FILE "aprs.csv"

void prototype::test_effective_resolution(learn_correct::Params params) {
  using cv::Mat;
  using std::cerr;
  using std::endl;

  // metrics: pixels
  int width_px = 1360;
  int height_px = 1024;
  // metrics: meters
  double pixel_sep = params.pixel_sep;
  double width = width_px * pixel_sep;
  double height = height_px * pixel_sep;
  double focal_length = params.focal_length;

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
    int w = 256 / k;
    int h = 256 / k;
    // compute distance map at downscaled resolution
    Mat Dd = Mat::zeros(h, w, CV_32F);
    interp::distance_map(Dd, alt, pitch, roll, width, height, focal_length);
    // upscale using high-quality interpolation
    Mat Du;
    cv::resize(Dd, Du, D.size(), 0, 0, CV_INTER_LANCZOS4);
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
  log("ADDED %s") % inpath;
}

void out_flat_task(learn_correct::Params* params, illum::Lightfield* R, illum::Lightfield* G, illum::Lightfield* B, boost::mutex* mutex, string inpath) {
  using boost::algorithm::ends_with;
  log("START %s") % inpath;
  try {
    string lop = inpath;
    boost::to_lower(lop);
    cv::Mat bgr_LR;
    if(ends_with(lop,".png")) { // 8-bit color png?
      bgr_LR = imread(inpath); // read it
    } else if(ends_with(lop,".tif") || ends_with(lop,".tiff")) { // 16-bit raw TIFF?
      cv::Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH); // read full bit depth
      if(cfa_LR.empty())
	return;
      log("DEMOSAIC %s") % inpath;
      cv::Mat bgr_LR_16u = demosaic(cfa_LR, params->bayer_pattern) / 255; // debayer and scale intensity
      bgr_LR_16u.convertTo(bgr_LR, CV_8U);
    }
    if(bgr_LR.empty())
      return;
    // extract color channels
    std::vector<cv::Mat> channels;
    cv::split(bgr_LR, channels);
    { // protect frame average
      boost::lock_guard<boost::mutex> lock(*mutex);
      // BGR
      B->addImage(channels[0]);
      G->addImage(channels[1]);
      R->addImage(channels[2]);
    }
    log("ADDED %s") % inpath;
  } catch(std::runtime_error const &e) {
    log("ERROR adding %s: %s") % inpath % e.what();
  } catch(std::exception) {
    log("ERROR adding %s") % inpath;
  }
}

void prototype::test_flatness(learn_correct::Params params) {
  using std::cerr;
  // before any OpenCV operations are done, set global error flag
  cv::setBreakOnError(true);
  // ersatz logging setup
  using boost::format;
  cerr << nounitbuf;
  boost::mutex correctMutex;
  illum::Lightfield R;
  illum::Lightfield G;
  illum::Lightfield B;
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
  istream *csv_in = learn_correct::get_input(params);
  string line;
  while(getline(*csv_in,line)) { // read pathames from a file
    try {
      string outpath = line;
      if(fs::exists(outpath)) {
	io_service.post(boost::bind(out_flat_task, &params, &R, &G, &B, &correctMutex, outpath));
	log("QUEUED %s") % outpath;
      } else {
	log("WARNING: can't find %s") % outpath;
      }
    } catch(std::runtime_error const &e) {
      log("ERROR parsing input metadata: %s") % e.what();
    } catch(std::exception) {
      log("ERROR parsing input metadata");
    }
  }
  // destroy the work object to indicate that there are no more jobs
  work.reset();
  // now run all pending jobs to completion
  workers.join_all();

  cerr << "WRITING correct average R" << std::endl;
  imwrite("avg_correct_R.tiff",R.getAverage());
  cerr << "WRITING correct average G" << std::endl;
  imwrite("avg_correct_G.tiff",G.getAverage());
  cerr << "WRITING correct average B" << std::endl;
  imwrite("avg_correct_B.tiff",B.getAverage());
}
