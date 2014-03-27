#include <iostream>
#include <string>
#include <fstream>
#include <ios>
#include <boost/thread.hpp>
#include <boost/format.hpp>
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
#include "logging.hpp"

using std::string;

namespace fs = boost::filesystem;

using illum::MultiLightfield;
using learn_correct::Params;
using stereo::CameraPair;
using jlog::log;
using jlog::log_error;

using boost::format;
using boost::str;

#define RAD2DEG 57.29578

class WorkState {
private:
  boost::mutex mutex; // for protecting concurrent modification
  std::vector<string > learned; // all images in the learn set (learn phase)
  std::set<string > skip; // which images to skip (learn phase)
  void write_paramfile(fs::path paramfile) {
    // write Params to param file
    log("WRITING paramfile %s") % paramfile;
    std::ofstream pout(paramfile.string().c_str());
    if(!(pout << params)) // write parameters to parameter file
      throw std::runtime_error(str(format("failed to write parameter file to %s") % paramfile.c_str()));
    pout.close();
    log("WROTE paramfile %s") % paramfile;
  }
  void write_lightmap(fs::path outdir) {
    log("SAVING lightmap in %s ...") % outdir;
    model.save(outdir.string());
    log("SAVED lightmap in %s") % outdir;
  }
  void write_skipfile(fs::path skipfile) {
    // write "learned" CSV data to skip file
    log("WRITING skipfile %s") % skipfile;
    std::ofstream sout(skipfile.string().c_str());
    for(std::vector<std::string>::const_iterator i = learned.begin(); i != learned.end(); ++i) {
      sout << *i << std::endl;
    }
    sout.close();
    log("WROTE skipfile %s") % skipfile;
  }
  void read_skipfile(fs::path skipfile) {
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
    log("READING skipfile %s") % skipfile;
    std::ifstream sin(skipfile.string().c_str());
    string line;
    while(getline(sin,line)) {
      learned.push_back(line); // push CSV record to learned
      Tokenizer tok(line);
      string infile = *tok.begin();
      skip.insert(infile); // filename is first item in CSV record
    }
    log("READ skipfile %s") % skipfile;
  }
public:
  Params params;
  illum::MultiLightfield model; // the lightfield
  stereo::CameraPair cameras; // the camera metrics
  WorkState(Params p) {
    params = p;
    model = MultiLightfield(p.alt_spacing, p.focal_length, p.pixel_sep, p.camera_sep);
    cameras = CameraPair(p.camera_sep, p.focal_length, p.pixel_sep);
  }
  // checkpoint the current state of a learn process
  void checkpoint(string _outdir="") {
    // output directory (_outdir, if non-empty, overrides parameter)
    fs::path outdir(_outdir.empty() ? params.lightmap_dir : _outdir);
    // parameter file contains program options
    fs::path paramfile = outdir / "params.txt";
    // skipfile lists images in the lightmap along with alt / pitch / roll
    fs::path skipfile = outdir / "learned.csv";
    // make sure the directory exists
    if(params.create_directories)
      fs::create_directories(outdir);
    if(!fs::exists(outdir))
      throw std::runtime_error(str(format("lightmap directory %s does not exist") % outdir));
    // now write params
    write_paramfile(paramfile);
    // now write lightmap
    write_lightmap(outdir);
    // now write skipfile
    write_skipfile(skipfile);
  }
  int load(string _outdir="") {
    // output directory (_outdir, if non-empty, overrides parameter)
    fs::path outdir(_outdir.empty() ? params.lightmap_dir : _outdir);
    // now read lightmap
    log("LOADING lightmap from %s ...") % outdir;
    return model.load(outdir.string());
    log("LOADED lightmap from %s") % outdir;
  }
  // resume from an existing lightmap
  int resume(string _outdir="") {
    // output directory (_outdir, if non-empty, overrides parameter)
    fs::path outdir(_outdir.empty() ? params.lightmap_dir : _outdir);
    // skipfile lists images in the lightmap along with alt / pitch / roll
    fs::path skipfile = outdir / "learned.csv";
    // FIXME read parameters from param file
    // read skipfile, if there is one
    if(fs::exists(skipfile))
      read_skipfile(skipfile);
    // now read lightmap
    load(outdir.string());
  }
  void add_learned(string inpath, double alt, double pitch, double roll) {
    // convert pitch/roll back to degrees
    pitch *= RAD2DEG;
    roll *= RAD2DEG;
    { // protect learned list with mutex to prevent concurrent writes
      boost::lock_guard<boost::mutex> lock(mutex);
      learned.push_back(str(format("%s,%.2f,%.2f,%.2f") % inpath % alt % pitch % roll));
    }
  }
  int n_learned() {
    return learned.size();
  }
  bool should_skip(string inpath) {
    return skip.count(inpath) > 0;
  }
};

double compute_missing_alt(WorkState* state, double alt, cv::Mat cfa_LR, std::string inpath) {
  using stereo::align;
  using cv::Mat;
  // if altitude is good, don't recompute it
  if(!state->params.alt_from_parallax && alt > 0 && alt < MAX_ALTITUDE)
    return alt;
  // compute from parallax
  // pull green channel
  Mat G;
  if(state->params.bayer_pattern[0]=='g') {
    cfa_channel(cfa_LR, G, 0, 0);
  } else {
    cfa_channel(cfa_LR, G, 1, 0);
  }
  // compute pixel offset
  int x = align(G, state->params.parallax_template_size) * 2;
  if(x <= 0)
    throw std::runtime_error(str(format("unable to compute altitude from parallax for %s") % inpath));
  // convert to meters
  alt = state->cameras.xoff2alt(x);
  // log what just happened
  log("PARALLAX altitude of %s is %.2f") % inpath % alt;
  return alt;
}

// the learn task adds an image to a multilightfield model
void learn_task(WorkState* state, string inpath, double alt, double pitch, double roll) {
  using cv::Mat;
  // get the input pathname
  try  {
    log("START LEARN %s %.2f,%.2f,%.2f") % inpath % alt % pitch % roll;
    // read the image (this can be done in parallel)
    Mat cfa_LR = cv::imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
    if(!cfa_LR.data)
      throw std::runtime_error(str(format("unable to read image file: %s") % inpath));
    if(cfa_LR.type() != CV_16U)
      throw std::runtime_error(str(format("image is not 16-bit grayscale: %s") % inpath));
    log("READ %s") % inpath;
    // if altitude is out of range, compute from parallax
    alt = compute_missing_alt(state, alt, cfa_LR, inpath);
    state->model.addImage(cfa_LR, alt, pitch, roll);
    state->add_learned(inpath, alt, pitch, roll);
    log("LEARNED %s") % inpath;
  } catch(std::runtime_error const &e) {
    log_error("ERROR learning %s: %s") % inpath % e.what();
  } catch(std::exception) {
    log_error("ERROR learning %s") % inpath;
  }
}

// the correct task corrects images
void correct_task(WorkState* state, string inpath, double alt, double pitch, double roll, string outpath) {
  using cv::Mat;
  using boost::algorithm::ends_with;
  try {
    Params* params = &state->params;
    log("START CORRECT %s %.2f,%.2f,%.2f") % inpath % alt % pitch % roll;
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
    // if we're skipping existing images, check once again for existence
    if(params->skip_existing && fs::exists(outp)) {
      log("SKIPPING %s because %s exists") % inpath % outpath;
      return;
    }
    // proceed
    Mat cfa_LR = cv::imread(inpath, CV_LOAD_IMAGE_ANYDEPTH); // read input image
    if(!cfa_LR.data)
      throw std::runtime_error("no image data");
    if(cfa_LR.type() != CV_16U)
      throw std::runtime_error("image is not 16-bit grayscale");
    // if altitude is out of range, compute from parallax
    alt = compute_missing_alt(state, alt, cfa_LR, inpath);
    // get the average
    Mat average = Mat::zeros(cfa_LR.size(), CV_32F);
    state->model.getAverage(average, alt, pitch, roll);
    // now smooth the average
    int h = average.size().height;
    int w = average.size().width;
    if(params->stereo) {
      Mat left = Mat(average,cv::Rect(0,0,w/2,h));
      Mat right = Mat(average,cv::Rect(w/2,0,w/2,h));
      cfa_smooth(left,left,params->lightmap_smoothing);
      cfa_smooth(right,right,params->lightmap_smoothing);
    } else {
      cfa_smooth(cfa_LR,cfa_LR,params->lightmap_smoothing);
    }
    log("SMOOTHED lightmap for %s") % inpath;
    illum::correct(cfa_LR, cfa_LR, average); // correct it
    log("DEMOSAICING %s") % inpath;
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
    log("SAVING corrected image to %s") % outpath;
    if(!imwrite(outpath, rgb_LR_8u))
      throw std::runtime_error(str(format("unable to write output image to %s") % outpath));
    log("CORRECTED %s") % inpath;
  } catch(std::runtime_error const &e) {
    log_error("ERROR correcting %s: %s") % inpath % e.what();
  } catch(std::exception) {
    log_error("ERROR correcting %s") % inpath;
  }
}

std::istream* learn_correct::get_input(learn_correct::Params p) {
  if(p.input == "-") { // use stdin
    return &std::cin;
  } else {
    return new std::ifstream(p.input.c_str());
  }
}

void do_learn_correct(learn_correct::Params p, bool learn, bool correct) {
  using learn_correct::Task;
  // before any OpenCV operations are done, set global error flag
  cv::setBreakOnError(true);
  WorkState state(p);
  if(learn && !p.update.empty()) { // updating?
    state.resume(p.update);
  } else if(learn && !p.update.empty()) {
    state.checkpoint();
  } else if(correct) {
    int loaded = state.load();
    if(!learn && !loaded)
      throw std::runtime_error(str(format("lightmap in %s is empty, cannot correct without training") % p.lightmap_dir));
  }
  log("READY to start processing");
  // now do a chunk of work, checkpoint, and continue
  int n_todo = 0;
  while(n_todo <= 0) {
    // post all work
    boost::asio::io_service io_service;
    boost::thread_group workers;
    // start up the work threads
    // use the work object to keep threads alive before jobs are posted
    // use auto_ptr so we can indicate that no more jobs will be posted
    std::auto_ptr<boost::asio::io_service::work> work(new boost::asio::io_service::work(io_service));
    // create the thread pool
    for(int i = 0; i < p.n_threads; i++) {
      workers.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
    }
    // now read input lines and post jobs
    std::istream* csv_in = get_input(p);
    n_todo = p.batch_size; // how many images to process in this batch
    string line;
    int n_learned_then = state.n_learned();
    while(getline(*csv_in,line) && (!learn || n_todo--)) { // read pathames from a file
      try {
	// parse the input line and turn it into a Task object
	Task task = Task(line);
	// check that the task is valid
	task.validate();
	if(learn) { // if learning
	  // is the inpath on the skip list?
	  if(!state.should_skip(task.inpath)) {
	    // push a learn task on the queue
	    io_service.post(boost::bind(learn_task, &state, task.inpath, task.alt, task.pitch, task.roll));
	    log("QUEUED LEARN %s") % task.inpath;
	  } else {
	    log("SKIPPED LEARN %s") % task.inpath;
	  }
	}
	if(correct) { // if correcting
	  if(p.skip_existing && fs::exists(task.outpath)) {
	    log("SKIPPED CORRECT %s") % task.outpath;
	  } else {
	    // push a correct task on the queue
	    io_service.post(boost::bind(correct_task, &state, task.inpath, task.alt, task.pitch, task.roll, task.outpath));
	    log("QUEUED CORRECT %s") % task.inpath;
	  }
	}
      } catch(std::runtime_error const &e) {
	log_error("ERROR parsing input metadata: %s: last line read was '%s'") % e.what() % line;
      } catch(std::exception) {
	log_error("ERROR parsing input metadata: last line read was '%s'") % line;
      }
    }
    // destroy the work object to indicate that there are no more jobs
    work.reset();
    // now run all pending jobs to completion
    workers.join_all();

    int n_learned_now = state.n_learned();
    // we know output directory already exists and can be written to
    if(learn && n_todo > 0) { // final output? (did less than the batch size?)
      state.checkpoint(p.lightmap_dir);
    } else if(learn && n_todo < p.batch_size && n_learned_now > n_learned_then) {
      // shuffle directories around
      fs::path outd(p.lightmap_dir);
      fs::path latest_checkpoint = outd / "latest";
      fs::path prev_checkpoint = outd / "prev_checkpoint";
      fs::path new_checkpoint = outd / "new_checkpoint";
      // first write the new checkpoint
      // if the directory exists, empty it first
      if(fs::exists(new_checkpoint)) {
	fs::remove_all(new_checkpoint);
      }
      // now write the new checkpoint
      state.checkpoint(new_checkpoint.string());
      // move "latest" checkpoint safely out of the way
      if(fs::exists(latest_checkpoint)) {
	log("CHECKPOINT moving old checkpoint to %s") % prev_checkpoint;
	fs::rename(latest_checkpoint, prev_checkpoint);
      }
      // now move it into place
      fs::rename(new_checkpoint, latest_checkpoint);
      log("CHECKPOINT moved to %s") % latest_checkpoint;
      // now remove previous checkpoint
      if(fs::exists(prev_checkpoint)) {
	log("CHECKPOINT deleting older %s ...") % prev_checkpoint;
	fs::remove_all(prev_checkpoint);
	log("CHECKPOINT deleted older %s") % prev_checkpoint;
      }
    }
  }// move on to next chunk
  log("COMPLETE");
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
