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
void learn_task(MultiLightfield<int> *model, string inpath, int alt) {
  static boost::mutex mutex; // shared lock for lightfield (is static sufficient?)
  // get the input pathname
  cout << "POPPED " << inpath << " " << alt << endl;
  // read the image (this can be done in parallel)
  Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
  cout << "Read " << inpath << endl;
  { // now lock the lightfield just long enough to add the image
    boost::lock_guard<boost::mutex> lock(mutex);
    model->addImage(cfa_LR, alt);
  }
  cout << "Added " << inpath << endl;
}

// the correct task corrects images
void correct_task(MultiLightfield<int> *model, string inpath, int alt, string outpath) {
  cout << "POPPED " << inpath << " " << alt << endl;
  Mat average = model->getAverage(alt);
  // now smooth the average
  int h = average.size().height;
  int w = average.size().width;
  Mat left = Mat(average,Rect(0,0,w/2,h));
  Mat right = Mat(average,Rect(w/2,0,w/2,h));
  cfa_smooth(left,left,31);
  cfa_smooth(right,right,31);
  cout << "SMOOTHED lightmap" << endl;
  Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH); // read input image
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
  MultiLightfield<int> model(100, 300, 10); // lightfield
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
    int alt = (int)(atof(fields.back().c_str()) * 100); // altitude in cm
    io_service.post(boost::bind(learn_task, &model, inpath, alt));
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
  MultiLightfield<int> model(100, 300, 10);
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
      int alt = (int)(atof(fields.back().c_str()) * 100);
      stringstream outpaths;
      string outpath;
      outpaths << OUT_DIR << "/correct" << count << ".tiff";
      outpath = outpaths.str();
      io_service.post(boost::bind(correct_task, &model, inpath, alt, outpath));
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

void prototype::test_alt_pitch_roll() {
  using namespace std;
  float focal_length_px = 2764.0; // value taken from JRock's code
  float pixel_sep = 0.0000065; // value taken from JHowland's code
  float focal_length_m = focal_length_px * pixel_sep;
  // V4 image metrics
  int width = 1360;
  int height = 1024;
  for(float alt = 0.5; alt < 4; alt += 0.25) {
    for(float pitch = -10; pitch <= 10; pitch += 5) {
      for(float roll = 20; roll >= -20; roll -= 5) {
	// JRock uses 3.14 as an approximation of Pi, so I'll use that too
	float pitch_rad = (3.14 * pitch / 180.0);
	float roll_rad = (3.14 * roll / 180.0);
	// compute both maps
	cv::Mat jmf_apr = interp::alt_pitch_roll(alt, pitch_rad, roll_rad, width, height, width, height, focal_length_m, pixel_sep);
	cv::Mat jrock_apr = jrock_calculate_altitude_map(alt, pitch, roll, width, height, focal_length_px);
	// convert jmf output to double precision
	cv::Mat jmf_apr_64;
	jmf_apr.convertTo(jmf_apr_64, CV_64F); // FIMXE was jmf_apr
	// compute absolute difference between them
	cv::Mat diff(width, height, CV_64F);
	cv::absdiff(jmf_apr_64, jrock_apr, diff);
	// now compute mean of absolute difference
	cv::Scalar savg_diff = cv::mean(diff);
	double avg_diff = savg_diff[0];
	// if the avg diff is over 1cm then there's probably a problem
	// and we should write out the matrices as images and exit
	if(avg_diff > 0.01) {
	  cv::Mat out(height, width*2, CV_8U);
	  // arbitrarily scale from substrate to alt*2
	  jmf_apr = (jmf_apr / (alt*2)) * 255;
	  jrock_apr = (jrock_apr / (alt*2)) * 255;
	  cv::Mat left = cv::Mat(out, cv::Rect(0, 0, width, height));
	  cv::Mat right = cv::Mat(out, cv::Rect(width, 0, width, height));
	  jmf_apr.convertTo(left, CV_8U);
	  jrock_apr.convertTo(right, CV_8U);
	  cout << "FAIL at altitude " << alt << "m, pitch " << pitch << "deg, roll " << roll << "deg" << endl;
	  imwrite("too_different.png",out);
	  exit(-1);
	}
      }
    }
    cout << "PASS altitude " << alt << "m" << endl;
  }
}

void prototype::test_distance_map() {
  using cv::Mat;
  double alt = 0.7;
  double pitch = M_PI * -12 / 180.0;
  double roll = M_PI * -33 / 180.0;
  double pixel_sep = 0.0000065;
  double width = 1001 * pixel_sep;
  double height = 1001 * pixel_sep;
  double focal_length = 0.012;

  Mat D = Mat::zeros(50, 50, CV_32F);
  interp::distance_map(D, alt, pitch, roll, width, height, focal_length);

  double minD, maxD;
  cv::minMaxLoc(D, &minD, &maxD);

  std::cout << minD << "," << maxD << endl;

  double delta = 0.1;

  Mat W = Mat::zeros(D.size(), CV_32F);
  for(int i = 0; i < 12; i++) {
    interp::dist_weight(D, W, delta, i);
    stringstream outpaths;
    string outpath;
    outpaths << "weight_" << i << ".tiff";
    W *= 256;
    Mat W_16u;
    W.convertTo(W_16u, CV_16U);
    imwrite(outpaths.str(), W);
  }
}
