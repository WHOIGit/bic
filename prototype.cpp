#include <iostream>
#include <string>
#include <fstream>
#include <ios>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/format.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "learn_correct.hpp"
#include "prototype.hpp"
#include "demosaic.hpp"
#include "illumination.hpp"
#include "interpolation.hpp"
#include "stereo.hpp"
#include "logging.hpp"

namespace fs = boost::filesystem;

using namespace std;
using namespace cv;

using jlog::log;
using jlog::log_error;

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
  log("START %s") % inpath;
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
    log_error("ERROR adding %s: %s") % inpath % e.what();
  } catch(std::exception) {
    log_error("ERROR adding %s") % inpath;
  }
}

void prototype::test_flatness(learn_correct::Params params) {
  using std::cerr;
  using std::endl;
  using cv::Mat;
  // before any OpenCV operations are done, set global error flag
  cv::setBreakOnError(true);
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
      boost::algorithm::trim_right(line);
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

  Mat Ra = R.getAverage();
  Mat Ga = G.getAverage();
  Mat Ba = B.getAverage();

  cerr << "WRITING correct average R" << std::endl;
  imwrite("avg_correct_R.tiff",Ra);
  cerr << "WRITING correct average G" << std::endl;
  imwrite("avg_correct_G.tiff",Ga);
  cerr << "WRITING correct average B" << std::endl;
  imwrite("avg_correct_B.tiff",Ba);

  std::vector<cv::Mat> bgr;
  bgr.push_back(Ba);
  bgr.push_back(Ga);
  bgr.push_back(Ra);
  cv::Mat BGR;
  cv::merge(bgr, BGR);
  cerr << "WRITING correct average BGR" << std::endl;
  imwrite("avg_correct.jpg",BGR);
}

void prototype::test_dm() {
  using cv::Mat;
  using std::cout;
  using std::endl;
  Mat image(1024, 2720, CV_32F);
  double alt=1.74;
  double pitch=0;
  double roll=0;
  stereo::CameraPair cameras(0.235, 0.012, 0.00000645);
  //
  alt = 1.45;
  cout << cameras.alt2xoff(alt) << endl;
  cout << cameras.xoff2alt(cameras.alt2xoff(alt)) << endl;
  // compute distance map
  int width = image.size().width;
  int height = image.size().height;
  double width_m = width * cameras.pixel_sep;
  double height_m = height * cameras.pixel_sep;
  Mat stereo(1024, 2720, CV_32F);
  Mat mono(1024, 2720, CV_32F);
  Mat left = Mat(stereo,cv::Rect(0,0,width/2,height));
  Mat right = Mat(stereo,cv::Rect(width/2,0,width/2,height));
  // now compute center of overlap region in sensor coordinates
  double xoff = cameras.alt2xoff(alt);
  double xoff_m = xoff * cameras.pixel_sep;
  double cx_R_px = width/2 - xoff/2; // in pixels in right frame coordaintes
  double cx_L_px = width/2 - cx_R_px; // in pixels in left frame coordinates
  double cx_R = cx_R_px * cameras.pixel_sep; // in meters
  double cx_L = cx_L_px * cameras.pixel_sep; // in meters
  cx_L = -width_m/4 + xoff_m/2;
  cx_R = width_m/4 - xoff_m/2;
  interp::distance_map(left, alt, pitch, roll, width_m/2, height_m, cameras.focal_length, cx_L);
  interp::distance_map(right, alt, pitch, roll, width_m/2, height_m, cameras.focal_length, cx_R);
  // for now, compute a distance map that in the case of stereo pairs spans the whole image
  interp::distance_map(mono, alt, pitch, roll, width_m, height_m, cameras.focal_length);
  cv::imwrite("dm_stereo.jpg",stereo * 100);
  cv::imwrite("dm_mono.jpg",mono * 100);
}

void prototype::afp(learn_correct::Params p) {
  // compute alt from parallax for a single image
  string inpath = p.input; // input must be pathname to 16-bit RAW image
  Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH); // read input image
  if(cfa_LR.empty()) {
    throw runtime_error("empty image");
  }
  stereo::CameraPair cameras(p.camera_sep, p.focal_length, p.pixel_sep);
  Mat y_LR;
  cfa_channel(cfa_LR, y_LR, 1, 0); // FIXME hardcoded
  int xoff = stereo::align(y_LR);
  double alt = cameras.xoff2alt(xoff*2);
  cout << xoff << "px, " << alt << "m" << endl;
}

void prototype::test_interpolation(learn_correct::Params p) {
  using boost::format;
  using boost::str;
  illum::MultiLightfield model(p.alt_spacing); // the lightfield
  //stereo::CameraPair cameras; // the camera metrics
  //cameras = CameraPair(p.camera_sep, p.focal_length, p.pixel_sep);
  // now read lightmap
  log_error("LOADING lightmap from %s ...") % p.lightmap_dir;
  int loaded = model.load(p.lightmap_dir);
  log_error("LOADED %d lightmap slices from %s") % loaded % p.lightmap_dir;
  // now generate averages
  for(double alt = 0; alt < 3; alt += 0.01) {
    try {
      Mat avg;
      model.getAverage(avg, alt);
      Scalar sm = cv::mean(avg);
      string msg = str(format("%.2f,%f") % alt % sm[0]);
      log_error(msg.c_str());
      cout << msg << endl;
    } catch(std::exception) {
      log_error("no images in %.2f bin") % alt;
    }
  }
}

void prototype::get_slice(learn_correct::Params p) {
  using boost::str;
  illum::Lightfield slice;
  //stereo::CameraPair cameras; // the camera metrics
  //cameras = CameraPair(p.camera_sep, p.focal_length, p.pixel_sep);
  // now read lightmap
  log_error("LOADING slice from %s") % p.input;
  slice.load(p.input);
  log_error("AVERAGING slice");
  Mat cfa_LR;
  slice.getAverage(cfa_LR);
  double minCount, maxCount;
  slice.minMaxCount(&minCount, &maxCount);
  log_error("min/max count = %f - %f") % minCount % maxCount;
  log_error("DEMOSAICING average");
  Mat bgr_LR = demosaic(cfa_LR, p.bayer_pattern) / 255; // lose the 16-bit bit depth
  string outfile = "slice_average.png";
  imwrite(outfile,bgr_LR);
  log_error("WRITING slice average to %s") % outfile;
}


void composite_overlap(Mat y_LR, int xoff, OutputArray _y_L, OutputArray _y_R) {
  int x2 = xoff / 2;
  Mat xMap, yMap;
  int w = y_LR.cols;
  int h = y_LR.rows;
  xMap.create(y_LR.size(), CV_32F);
  yMap.create(y_LR.size(), CV_32F);
  int c2 = w/2;
  for(int x = 0; x < w; x++) {
    for(int y = 0; y < h; y++) {
      yMap.at<float>(y,x) = y;
      if(x < c2) {
	xMap.at<float>(y,x) = x + c2 - x2;
      } else {
	xMap.at<float>(y,x) = -1;
      }
    }
  }
  Mat y_O;
  remap(y_LR,y_O,xMap,yMap,INTER_NEAREST);
  int o_width = w/2-x2;
  Rect overlap = Rect(x2,0,o_width,h);
  Mat y_L(y_LR, overlap);
  Mat y_R(y_O,overlap);
  _y_L.create(Size(o_width,h),y_LR.type());
  y_L.copyTo(_y_L);
  _y_R.create(Size(o_width,h),y_LR.type());
  y_R.copyTo(_y_R);
}

void avg_alt_task(learn_correct::Params* params, std::string inpath, double alt) {
  try  {
    Mat cfa_LR = cv::imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
    /*
    Mat bgr_LR = demosaic(cfa_LR, params->bayer_pattern);
    Scalar avg = cv::mean(bgr_LR);
    double avg_red = avg[2];
    double avg_blue = avg[0];
    double avg_green = avg[1];
    log("%.2f,%f,%f,%f") % alt % avg_red % avg_blue % avg_green;
    */
    // FIXME debug. compute offset and variance from a-f-p
    double afp_var;
    Mat G;
    if(params->bayer_pattern[0]=='g') {
      cfa_channel(cfa_LR, G, 0, 0);
    } else {
      cfa_channel(cfa_LR, G, 1, 0);
    }
    // compute pixel offset
    int xoff = stereo::align(G, params->parallax_template_size, &afp_var) * 2;
    if(xoff == 0)
      return;
    // report
    log("%s,%d,%.0f") % inpath % xoff % afp_var;
  } catch(std::exception) {
    log_error("ERROR for %s") % inpath;
  }
}

void prototype::avg_by_alt(learn_correct::Params params) {
  using std::cerr;
  using std::endl;
  using cv::Mat;
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
      learn_correct::Task task(line);
      io_service.post(boost::bind(avg_alt_task, &params, task.inpath, task.alt));
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

void prototype::redcyan(learn_correct::Params params) {
  using boost::algorithm::ends_with;
  using namespace cv;
  using stereo::align;
  using namespace std;
  Mat in_LR;
  if(ends_with(params.input,"tiff") || ends_with(params.input,"tif")) {
    in_LR = imread( params.input, CV_LOAD_IMAGE_ANYDEPTH );
  } else {
    in_LR = imread(params.input);
    resize(in_LR, in_LR, Size(0,0), 0.5, 0.5); // downscale to 0.5 size
  }

  // now, compute offset

  // compute from parallax

  Mat G;
  if(in_LR.channels() == 1) {   // if raw, pull green channel
    cout << "Extracting green channel of RAW image" << endl;
    if(params.bayer_pattern[0]=='g') {
      cfa_channel(in_LR, G, 0, 0);
    } else {
      cfa_channel(in_LR, G, 1, 0);
    }
  } else { // otherwise convert to gray
    // assume 8-bit
    cout << "Converting color image to grayscale" << endl;
    cvtColor(in_LR, G, CV_BGR2GRAY);
  }

  cout << "aligning ..." << endl;
  // compute pixel offset
  double var;
  int xoff = align(G, params.parallax_template_size,&var);
  int x2 = xoff;
  xoff *= 2;
  if(xoff <= 0) // bad alignment
    throw std::runtime_error("unable to compute altitude from parallax");
  // convert to meters
  double alt = (params.camera_sep * params.focal_length * 1.2) / (xoff * params.pixel_sep);

  cout << "offset = " << xoff << endl;
  cout << "alt = " << alt << endl;
  cout << "variance = " << var << endl;

  Mat y_L, y_R;
  composite_overlap(G,xoff,y_L,y_R);
  // brighten
  y_L *= 2;
  y_R *= 2;
  vector<Mat> ch;
  ch.push_back(y_R/2);
  ch.push_back(y_R/2);
  ch.push_back(y_L);
  Mat redcyan;
  merge(ch, redcyan);
  namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
  imshow( "Display Image", redcyan );

  waitKey(0);
}
