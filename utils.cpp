#include <string>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "utils.hpp"
#include "learn_correct.hpp"
#include "demosaic.hpp"
#include "stereo.hpp"
#include "logging.hpp"
#include "illumination.hpp"

namespace fs = boost::filesystem;

using namespace std;
using namespace cv;

using jlog::log;
using jlog::log_error;

using learn_correct::Params;
using stereo::CameraPair;

using boost::str;

void alt_task(Params* params, string inpath) {
  try {
    Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH); // read input image
    if(!cfa_LR.data)
      throw std::runtime_error("no image data");
    if(cfa_LR.type() != CV_16U)
      throw std::runtime_error("image is not 16-bit grayscale");
    // pull green channel
    // pull green channel
    Mat G;
    if(params->bayer_pattern[0]=='g') {
      cfa_channel(cfa_LR, G, 0, 0);
    } else {
      cfa_channel(cfa_LR, G, 1, 0);
    }
    // compute pixel offset
    int x = stereo::align(G, params->parallax_template_size) * 2;
    // set up a camera pair
    CameraPair cameras(params->camera_sep, params->focal_length, params->pixel_sep);
    double alt = cameras.xoff2alt(x);
    log("%s,%.2f") % inpath % alt;
  } catch(std::runtime_error const &e) {
    log_error("ERROR %s: %s") % inpath % e.what();
  } catch(std::exception) {
    log_error("ERROR %s") % inpath;
  }
}

void utils::alt_from_stereo(Params params) {
  // before any OpenCV operations are done, set global error flag
  cv::setBreakOnError(true);
  // post all work
  boost::asio::io_service io_service;
  boost::thread_group workers;
  // start up the work threads
  // use the work object to keep threads alive before jobs are posted
  // use auto_ptr so we can indicate that no more jobs will be posted
  auto_ptr<boost::asio::io_service::work> work(new boost::asio::io_service::work(io_service));
  // create the thread pool
  for(int i = 0; i < params.n_threads; i++) {
    workers.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
  }
  // post jobs
  istream *csv_in = learn_correct::get_input(params);
  string line;
  while(getline(*csv_in,line)) { // read pathames from a file
    try {
      boost::algorithm::trim_right(line);
      string inpath = line;
      if(fs::exists(inpath)) {
	io_service.post(boost::bind(alt_task, &params, inpath));
      } else {
	log_error("WARNING: can't find %s") % inpath;
      }
    } catch(std::runtime_error const &e) {
      log_error("ERROR parsing input metadata: %s") % e.what();
    } catch(std::exception) {
      log_error("ERROR parsing input metadata");
    }
  }
  // destroy the work object to indicate that there are no more jobs
  work.reset();
  // now run all pending jobs to completion
  workers.join_all();
}

void utils::view_raw(Params params) {
  using boost::algorithm::ends_with;
  using namespace cv;
  using namespace std;
  string terst = params.input;
  boost::to_lower(terst);
  assert(ends_with(terst,"tiff") || ends_with(terst,"tif"));
  Mat y_LR = imread( params.input, CV_LOAD_IMAGE_ANYDEPTH );
  // brighten
  double avg = mean(y_LR)[0];
  y_LR *= 32768.0 / avg;
  // demosaic
  Mat bgr_LR = demosaic_thumb_lq(y_LR, params.bayer_pattern);
  // display
  namedWindow(params.input.c_str(), CV_WINDOW_AUTOSIZE );
  imshow(params.input.c_str(), bgr_LR );
  waitKey(0);
}

void utils::view_xeye(Params params) {
  // assumes -i is a color left/right image
  Mat in_LR = imread(params.input);
  Mat y_LR;
  cvtColor(in_LR, y_LR, CV_BGR2GRAY);
  int xoff = stereo::align(y_LR);
  //int xoff = 0;
  Mat X = stereo::xeye(in_LR, xoff);
  // resize
  cv::resize(X, X, cv::Size(0,0), 0.66, 0.66);
  namedWindow(params.input.c_str(), CV_WINDOW_AUTOSIZE );
  imshow(params.input.c_str(), X );
  waitKey(0);
}

void utils::thumb_lightmap(Params params) {
  illum::MultiLightfield lightmap(params.alt_spacing);
  if(params.lightmap_dir.empty())
    throw std::runtime_error("no lightmap directory specified");
  log("LOADING lightmap ...");
  lightmap.load(params.lightmap_dir);
  for(double alt = 0; alt < 1000; alt += params.alt_spacing) {
    try {
      Mat raw_avg;
      lightmap.getAverage(raw_avg, alt);
      // now do a quick-and-dirty conversion to color
      Mat bgr_avg = demosaic_thumb_lq(raw_avg, params.bayer_pattern);
      // now convert to 8u
      bgr_avg *= 255.0 / 65535.0;
      fs::path p(params.lightmap_dir);
      p /= str(boost::format("thumb_slice_%.02f.jpg") % alt);
      log("SAVING thumbnail %s") % p.string();
      imwrite(p.string(), bgr_avg);
    } catch(std::exception) {
      // silently ignore
    }
  }
}

void utils::side_by_side(Params params) {
  Mat in_LR = imread(params.input);
  Mat y_LR;
  cvtColor(in_LR, y_LR, CV_BGR2GRAY);
  int xoff = stereo::align(y_LR);
  Mat sbs = stereo::sideBySide(in_LR, params.resolution_x, params.resolution_y, xoff);
  namedWindow(params.input.c_str(), CV_WINDOW_AUTOSIZE );
  imshow(params.input.c_str(), sbs );
  waitKey(0);
}
