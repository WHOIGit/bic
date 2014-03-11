#include <iostream>
#include <string>
#include <fstream>
#include <ios>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>

#include "learn_correct.hpp"
#include "prototype.hpp"
#include "demosaic.hpp"
#include "illumination.hpp"
#include "interpolation.hpp"

using namespace std;
using namespace cv;

using illum::MultiLightfield;
using learn_correct::Params;

// the learn task adds an image to a multilightfield model
void learn_task(Params *params, MultiLightfield *model, string inpath, double alt, double pitch, double roll) {
  using boost::format;
  cerr << nounitbuf;
  // get the input pathname
  try  {
    cerr << format("POPPED %s %.2f,%.2f,%.2f") % inpath % alt % pitch % roll << endl;
    // read the image (this can be done in parallel)
    Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
    if(!cfa_LR.data)
      throw std::runtime_error("unable to read image file");
    if(cfa_LR.type() != CV_16U)
      throw std::runtime_error("image is not 16-bit grayscale");
    cerr << format("READ %s") % inpath << endl;
    model->addImage(cfa_LR, alt, pitch, roll);
    cerr << format("ADDED %s") % inpath << endl;
  } catch(std::runtime_error const &e) {
    cerr << "ERROR learning " << inpath << ": " << e.what() << endl;
  } catch(std::exception) {
    cerr << "ERROR learning " << inpath << endl;
  }
}

// the correct task corrects images
void correct_task(Params *params, MultiLightfield *model, string inpath, double alt, double pitch, double roll, string outpath) {
  using boost::format;
  cerr << nounitbuf;
  try {
    cerr << format("POPPED %s %.2f,%.2f,%.2f") % inpath % alt % pitch % roll << endl;
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
    cfa_smooth(left,left,params->lightmap_smoothing);
    cfa_smooth(right,right,params->lightmap_smoothing);
    cerr << "SMOOTHED lightmap" << endl;
    illum::correct(cfa_LR, cfa_LR, average); // correct it
    cerr << format("DEMOSAICING %s") % inpath << endl;
    // demosaic it
    Mat rgb_LR = demosaic(cfa_LR,params->bayer_pattern);
    outpath = outpath + ".png";
    cerr << format("SAVE RGB to %s") % outpath << endl;
    // brightness and contrast parameters
    double max = params->max_brightness;
    double min = params->min_brightness;
    // adjust brightness/contrast and save as 8-bit png
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
void learn_correct::learn(learn_correct::Params p) {
  // before any OpenCV operations are done, set global error flag
  cv::setBreakOnError(true);
  // ersatz logging setup
  using boost::format;
  cerr << nounitbuf;
  // construct an empty lightfield model
  illum::MultiLightfield model(p.alt_spacing, p.focal_length, p.pixel_sep);
  // post all work
  boost::asio::io_service io_service;
  boost::thread_group workers;
  // start up the work threads
  // use the work object to keep threads alive before jobs are posted
  // use auto_ptr so we can indicate that no more jobs will be posted
  auto_ptr<boost::asio::io_service::work> work(new boost::asio::io_service::work(io_service));
  // create the thread pool
  for(int i = 0; i < p.n_threads; i++) {
    workers.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
  }
  // now read input lines and post jobs
  ifstream inpaths(PATH_FILE); // FIXME hardcoded CSV file pathname
  string line;
  while(getline(inpaths,line)) { // read pathames from a file
    try {
      Task task = Task(line);
      task.validate();
      io_service.post(boost::bind(learn_task, &p, &model, task.inpath, task.alt, task.pitch, task.roll));
      cerr << format("PUSHED LEARN %s") % task.inpath << endl;
    } catch(std::runtime_error const &e) {
      cerr << format("ERROR parsing input metadata: %s") % e.what() << endl;
    } catch(std::exception) {
      cerr << "ERROR parsing input metadata" << endl;
    }
  }
  // destroy the work object to indicate that there are no more jobs
  work.reset();
  // now run all pending jobs to completion
  workers.join_all();
  cerr << "SUCCESS learn phase" << endl;

  model.save(p.lightmap_dir);
  cerr << "SAVED model" << endl;
}

void learn_correct::correct(learn_correct::Params p) {
  // before any OpenCV operations are done, set global error flag
  cv::setBreakOnError(true);
  using boost::format;
  cerr << nounitbuf;
  // load model
  cerr << "LOADING model..." << endl;
  illum::MultiLightfield model;
  model.load(p.lightmap_dir);
  cerr << "LOADED model" << endl;

  // post all work
  boost::asio::io_service io_service;
  // use the work object to keep threads alive before jobs are posted
  // use auto_ptr so we can indicate that no more jobs will be posted
  auto_ptr<boost::asio::io_service::work> work(new boost::asio::io_service::work(io_service));
  // start a thread pool
  boost::thread_group workers;
  for(int i = 0; i < p.n_threads; i++) {
    workers.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
  }
  // post jobs
  ifstream inpaths(PATH_FILE);  // FIXME hardcoded CSV file pathname
  string line;
  while(getline(inpaths,line)) { // read pathames from a file
    try {
      Task task = Task(line);
      task.validate();
      io_service.post(boost::bind(correct_task, &p, &model, task.inpath, task.alt, task.pitch, task.roll, task.outpath));
      cerr << format("PUSHED CORRECT %s") % task.inpath << endl;
    } catch(std::runtime_error const &e) {
      cerr << format("ERROR parsing input metadata: %s") % e.what() << endl;
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
