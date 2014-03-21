#include <iostream>
#include <string>
#include <fstream>
#include <ios>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "learn_correct.hpp"
#include "stereo.hpp"
#include "prototype.hpp"
#include "demosaic.hpp"
#include "illumination.hpp"
#include "interpolation.hpp"

namespace fs = boost::filesystem;

using namespace std;
using namespace cv;

using illum::MultiLightfield;
using learn_correct::Params;

double compute_missing_alt(Params *params, double alt, cv::Mat cfa_LR, std::string inpath) {
  using stereo::align;
  using cv::Mat;
  using boost::format;
  // if altitude is good, don't recompute it
  if(alt > 0 && alt < MAX_ALTITUDE) {
    return alt;
  }
  // if images aren't stereo, we can't be doing this
  if(!params->stereo)
    throw std::runtime_error("cannot compute parallax from single-camera image");
  // compute from parallax
  // pull green channel
  Mat G;
  if(params->bayer_pattern[0]=='g') {
    cfa_channel(cfa_LR, G, 0, 0);
  } else {
    cfa_channel(cfa_LR, G, 1, 0);
  }
  // compute pixel offset
  int x = align(G, params->parallax_template_size) * 2;
  if(x <= 0) // bad alignment
    throw std::runtime_error("unable to compute altitude from parallax");
  // convert to meters
  alt = (params->camera_sep * params->focal_length * H2O_ADJUSTMENT) / (x * params->pixel_sep);
  // log what just happened
  cerr << format("PARALLAX altitude of %s is %.2f") % inpath % alt << endl;
  return alt;
}

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
    // if altitude is out of range, compute from parallax
    alt = compute_missing_alt(params, alt, cfa_LR, inpath);
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
  using namespace std;
  using boost::format;
  using boost::algorithm::ends_with;
  cerr << nounitbuf;
  try {
    cerr << format("POPPED %s %.2f,%.2f,%.2f") % inpath % alt % pitch % roll << endl;
    // make sure output path ends with ".png"
    string lop = outpath;
    boost::to_lower(lop);
    if(!ends_with(lop,".png"))
      outpath = outpath + ".png";
    // first, make sure we can write the output file
    fs::path outp(outpath);
    fs::path outdir = outp.parent_path();
    // now create output directory if necessary
    if(params->create_directories)
      fs::create_directories(outdir);
    // proceed
    Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH); // read input image
    if(!cfa_LR.data)
      throw std::runtime_error("no image data");
    if(cfa_LR.type() != CV_16U)
      throw std::runtime_error("image is not 16-bit grayscale");
    // if altitude is out of range, compute from parallax
    alt = compute_missing_alt(params, alt, cfa_LR, inpath);
    // get the average
    Mat average = Mat::zeros(cfa_LR.size(), CV_32F);
    model->getAverage(average, alt, pitch, roll);
    // now smooth the average
    int h = average.size().height;
    int w = average.size().width;
    if(params->stereo) {
      Mat left = Mat(average,Rect(0,0,w/2,h));
      Mat right = Mat(average,Rect(w/2,0,w/2,h));
      cfa_smooth(left,left,params->lightmap_smoothing);
      cfa_smooth(right,right,params->lightmap_smoothing);
    } else {
      cfa_smooth(cfa_LR,cfa_LR,params->lightmap_smoothing);
    }
    cerr << "SMOOTHED lightmap" << endl;
    illum::correct(cfa_LR, cfa_LR, average); // correct it
    cerr << format("DEMOSAICING %s") % inpath << endl;
    // demosaic it
    Mat rgb_LR = demosaic(cfa_LR,params->bayer_pattern);
    // brightness and contrast parameters
    double max = params->max_brightness;
    double min = params->min_brightness;
    // adjust brightness/contrast and save as 8-bit png
    Mat rgb_LR_8u;
    rgb_LR = rgb_LR * (255.0 / (65535.0 * (max - min))) - (min * 255.0);
    rgb_LR.convertTo(rgb_LR_8u, CV_8U);
    // now write the output image
    cerr << format("SAVE RGB to %s") % outpath << endl;
    if(!imwrite(outpath, rgb_LR_8u))
      throw std::runtime_error("unable to write output image");
  } catch(std::runtime_error const &e) {
    cerr << "ERROR correcting " << inpath << ": " << e.what() << endl;
  } catch(std::exception) {
    cerr << "ERROR correcting " << inpath << endl;
  }
}

std::istream* learn_correct::get_input(learn_correct::Params p) {
  if(p.input == "-") { // use stdin
    return &cin;
  } else {
    return new ifstream(p.input.c_str());
  }
}

void do_learn_correct(learn_correct::Params p, bool learn, bool correct) {
  using learn_correct::Task;
  // before any OpenCV operations are done, set global error flag
  cv::setBreakOnError(true);
  // ersatz logging setup
  using boost::format;
  cerr << nounitbuf;
  if(learn) {
    // before we begin, make sure we can write to the output directory by attempting
    // to write a parameter file
    fs::path outdir(p.lightmap_dir);
    if(p.create_directories)
      fs::create_directories(outdir);
    if(!fs::exists(outdir))
      throw std::runtime_error("output directory does not exist");
    fs::path paramfile = outdir / "params.txt";
    ofstream pout(paramfile.string().c_str());
    if(!(pout << p)) // write parameters to parameter file
      throw std::runtime_error("failed to write parameter file");
    pout.close();
  }
  // construct an empty lightfield model based on parameters
  illum::MultiLightfield model(p.alt_spacing, p.focal_length, p.pixel_sep);
  // load model
  if(correct || (learn && p.update)) {
    cerr << format("LOADING model from %s...") % p.lightmap_dir << endl;
    int loaded = model.load(p.lightmap_dir);
    if(!learn && !loaded)
      throw std::runtime_error("lightmap is empty, cannot correct without training");
    cerr << format("LOADED model from %s") % p.lightmap_dir << endl;
  }
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
  istream* csv_in = get_input(p);
  string line;
  while(getline(*csv_in,line)) { // read pathames from a file
    try {
      Task task = Task(line);
      task.validate();
      if(learn) {
	io_service.post(boost::bind(learn_task, &p, &model, task.inpath, task.alt, task.pitch, task.roll));
	cerr << format("PUSHED LEARN %s") % task.inpath << endl;
      }
      if(correct) {
	io_service.post(boost::bind(correct_task, &p, &model, task.inpath, task.alt, task.pitch, task.roll, task.outpath));
	cerr << format("PUSHED CORRECT %s") % task.inpath << endl;
      }
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
  cerr << "SUCCESS" << endl;

  // we know output directory already exists and can be written to
  if(learn) {
    cerr << format("SAVING model in %s...") % p.lightmap_dir << endl;
    model.save(p.lightmap_dir);
    cerr << format("SAVED model in %s") % p.lightmap_dir << endl;
  }
}

// learn phase
void learn_correct::learn(learn_correct::Params p) {
  do_learn_correct(p, true, false);
}

void learn_correct::correct(learn_correct::Params p) {
  do_learn_correct(p, false, true);
}

// adaptive (learn + correct)
void learn_correct::adaptive(learn_correct::Params p) {
  do_learn_correct(p, true, true);
}
