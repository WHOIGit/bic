#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/tokenizer.hpp>

#include "demosaic.hpp"
#include "illumination.hpp"

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
void learn_prototype() {
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

void correct_prototype() {
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
  // now run all work to completion
  io_service.run();
  cout << "SUCCESS correct phase" << endl;
}
