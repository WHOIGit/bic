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
  try  {
    cerr << "POPPED " << inpath << " " << alt << "," << pitch << "," << roll << endl;
    // read the image (this can be done in parallel)
    Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
    if(!cfa_LR.data)
      throw std::runtime_error("unable to read image file");
    if(cfa_LR.type() != CV_16U)
      throw std::runtime_error("image is not 16-bit grayscale");
    cerr << "Read " << inpath << endl;
    model->addImage(cfa_LR, alt, pitch, roll);
    cerr << "Added " << inpath << endl;
  } catch(std::runtime_error const &e) {
    cerr << "ERROR learning " << inpath << ": " << e.what() << endl;
  } catch(std::exception) {
    cerr << "ERROR learning " << inpath << endl;
  }
}

// the correct task corrects images
void correct_task(MultiLightfield *model, string inpath, double alt, double pitch, double roll, string outpath) {
  try {
    cerr << "POPPED " << inpath << " " << alt << "," << pitch << "," << roll << endl;
    Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH); // read input image
    if(!cfa_LR.data)
      throw std::runtime_error("no image data");
    if(cfa_LR.type() != CV_16U)
      throw std::runtime_error("image is not 16-bit grayscale");
    // get the average
    Mat average = Mat::zeros(cfa_LR.size(), CV_32F);
    model->getAverage(average, alt, pitch, roll);
    // now smooth the average
    int h = average.size().height;
    int w = average.size().width;
    Mat left = Mat(average,Rect(0,0,w/2,h));
    Mat right = Mat(average,Rect(w/2,0,w/2,h));
    // FIXME hardcoded param smoothing kernel size
    cfa_smooth(left,left,30); // testing even-to-odd conversion
    cfa_smooth(right,right,30);
    cerr << "SMOOTHED lightmap" << endl;
    illum::correct(cfa_LR, cfa_LR, average); // correct it
    cerr << "Demosaicing " << inpath << endl;
    // FIXME hardcoded param bayer pattern
    Mat rgb_LR = demosaic(cfa_LR,BAYER_PATTERN); // demosaic it
    /*stringstream outpath_exts;
      string outpath_ext;
      outpath_exts << outpath << ".png";
      outpath_ext = outpath_exts.str();*/
    outpath = outpath + ".png";
    cerr << "Saving RGB to " << outpath << endl;
    // FIXME hardcoded brightness parameters
    double max = 0.7;
    double min = 0.05;
    // scale brightness
    //rgb_LR /= max - min;
    //rgb_LR += min;
    // save as 8-bit png
    Mat rgb_LR_8u;
    rgb_LR = rgb_LR * (255.0 / (65535.0 * (max - min))) - (min * 255.0);
    rgb_LR.convertTo(rgb_LR_8u, CV_8U);
    imwrite(outpath, rgb_LR_8u);
  } catch(std::runtime_error const &e) {
    cerr << "ERROR correcting " << inpath << ": " << e.what() << endl;
  } catch(std::exception) {
    cerr << "ERROR correcting " << inpath << endl;
  }
}

// learn phase
void prototype::learn() {
  // before any OpenCV operations are done, set global error flag
  cv::setBreakOnError(true);
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
    try {
      Tokenizer tok(line);
      fields.assign(tok.begin(),tok.end());
      // FIXME tile
      int f=0;
      string inpath = fields.at(f++);
      string outpath = fields.at(f++);
      double alt = atof(fields.at(f++).c_str()); // altitude in m
      double pitch_deg = atof(fields.at(f++).c_str());
      double roll_deg = atof(fields.at(f++).c_str());
      double pitch = M_PI * pitch_deg / 180.0;
      double roll = M_PI * roll_deg / 180.0;
      io_service.post(boost::bind(learn_task, &model, inpath, alt, pitch, roll));
      cerr << "PUSHED " << inpath << endl;
    } catch(std::runtime_error const &e) {
      cerr << "ERROR parsing input metadata: " << e.what() << endl;
    } catch(std::exception) {
      cerr << "ERROR parsing input metadata" << endl;
    }
  }
  // destroy the work object to indicate that there are no more jobs
  work.reset();
  // now run all pending jobs to completion
  workers.join_all();
  cerr << "SUCCESS learn phase" << endl;

  model.save(OUT_DIR);
  cerr << "SAVED model" << endl;
}

void prototype::correct() {
  // before any OpenCV operations are done, set global error flag
  cv::setBreakOnError(true);
  // load model
  cerr << "LOADING model..." << endl;
  illum::MultiLightfield model;
  model.load(OUT_DIR);
  cerr << "LOADED model" << endl;

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
  while(getline(inpaths,line)) { // read pathames from a file
    try {
      Tokenizer tok(line);
      fields.assign(tok.begin(),tok.end());
      int f=0;
      string inpath = fields.at(f++);
      string outpath = fields.at(f++);
      double alt = atof(fields.at(f++).c_str()); // altitude in m
      double pitch_deg = atof(fields.at(f++).c_str());
      double roll_deg = atof(fields.at(f++).c_str());
      double pitch = M_PI * pitch_deg / 180.0;
      double roll = M_PI * roll_deg / 180.0;
      io_service.post(boost::bind(correct_task, &model, inpath, alt, pitch, roll, outpath));
      cerr << "PUSHED " << inpath << endl;
    } catch(std::runtime_error const &e) {
      cerr << "ERROR parsing input metadata: " << e.what() << endl;
    } catch(std::exception) {
      cerr << "ERROR parsing input metadata" << endl;
    }
  }
  // destroy the work object to indicate that there are no more jobs
  work.reset();
  // now run all pending jobs to completion
  workers.join_all();
  // we're done
  cerr << "SUCCESS correct phase" << endl;
}

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
  ifstream inpaths(PATH_FILE);
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
