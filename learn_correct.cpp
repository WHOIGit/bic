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
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>
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
      boost::algorithm::trim_right(line);
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
    cameras = CameraPair(p.camera_sep, p.focal_length, p.pixel_sep);
    model = MultiLightfield(p.alt_spacing, p.undertrain, p.overtrain);
    // model = illum::VehicleLightfield(p.alt_spacing, cameras)
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
    int loaded = model.load(outdir.string());
    log("LOADED %d lightmap slices from %s") % loaded % outdir;
    return loaded;
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
  // if images are not stereo, do not attempt to compute altitude from parallax
  if(!state->params.stereo)
    return alt;
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
    throw std::runtime_error(str(format("SKIPPING %s: unable to compute altitude from parallax") % inpath));
  // convert to meters
  alt = state->cameras.xoff2alt(x);
  // log what just happened
  log("PARALLAX altitude of %s is %.2f") % inpath % alt;
  return alt;
}

// read an image from a file and make sure it's 16-bit RAW
cv::Mat read_16u(string inpath) {
  cv::Mat img = cv::imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
  if(!img.data)
    throw std::runtime_error(str(format("ERROR: unable to read image file: %s") % inpath));
  if(img.type() != CV_16U)
    throw std::runtime_error(str(format("ERROR: image is not 16-bit grayscale: %s") % inpath));
  return img;
}

// learn one image
void learn_one(WorkState* state, cv::Mat cfa_LR, string inpath, double alt=0, double pitch=0, double roll=0) {
  if(!state->model.addImage(cfa_LR, alt)) {
    log("SKIPPING learn for %s - lightmap slice(s) overtrained");
  } else {
    state->add_learned(inpath, alt, pitch, roll);
  }
}

// the learn task adds an image to a multilightfield model
void learn_task(WorkState* state, string inpath, double alt, double pitch, double roll) {
  using cv::Mat;
  // get the input pathname
  try  {
    log("START LEARN %s %.2f,%.2f,%.2f") % inpath % alt % pitch % roll;
    // read the image (this can be done in parallel)
    Mat cfa_LR = read_16u(inpath);
    log("READ %s") % inpath;
    // determine altitude if necessary
    alt = compute_missing_alt(state, alt, cfa_LR, inpath);
    // now learn the image
    learn_one(state, cfa_LR, inpath, alt, pitch, roll);
    log("LEARNED %s") % inpath;
  } catch(std::runtime_error const &e) {
    log_error("DID NOT LEARN %s: %s") % inpath % e.what();
  } catch(std::exception) {
    log_error("ERROR learning %s") % inpath;
  }
}

// correct one image (and return corrected image)
cv::Mat correct_one(WorkState* state, cv::Mat cfa_LR, string inpath, double alt, double pitch, double roll) {
  using cv::Mat;
  Params* params = &state->params;
  // get the average
  Mat average;
  state->model.getAverage(average, alt);
  // now smooth the average
  int h = average.size().height;
  int w = average.size().width;
  if(params->stereo) {
    Mat left = Mat(average,cv::Rect(0,0,w/2,h));
    Mat right = Mat(average,cv::Rect(w/2,0,w/2,h));
    cfa_smooth(left,left,params->lightmap_smoothing);
    cfa_smooth(right,right,params->lightmap_smoothing);
  } else {
    cfa_smooth(average,average,params->lightmap_smoothing);
  }
  log("SMOOTHED lightmap for %s") % inpath;
  Mat corrected;
  illum::correct(cfa_LR, corrected, average); // correct it
  log("DEMOSAICING %s") % inpath;
  // demosaic it
  Mat rgb_LR = demosaic(corrected,params->bayer_pattern);
  // brightness and contrast parameters
  double max = params->max_brightness;
  double min = params->min_brightness;
  // adjust brightness/contrast and save as 8-bit png
  Mat rgb_LR_8u;
  rgb_LR = rgb_LR * (255.0 / (max - min)) - (min * 255.0);
  rgb_LR.convertTo(rgb_LR_8u, CV_8U);
  return rgb_LR_8u;
}

// return outpath if it is able and allowed to be written
string check_outpath(Params* params, string inpath, string outpath) {
  using boost::format;
  using boost::str;
  // first, make sure we can write the output file
  fs::path outp(outpath);
  // if we're skipping existing images, check once again for existence
  if(params->skip_existing && fs::exists(outp)) {
    throw std::runtime_error(str(format("SKIPPING %s because %s exists") % inpath % outpath));
  }
  return outpath;
}

// write a corrected image to its outpath
void write_corrected(Params* params, cv::Mat rgb_LR, string outpath) {
  // now create output directory if necessary
  fs::path outp(outpath);
  fs::path outdir = outp.parent_path();
  if(params->create_directories)
    fs::create_directories(outdir);
  // now write the output image
  log("SAVING corrected image to %s") % outpath;
  if(!imwrite(outpath, rgb_LR))
    throw std::runtime_error(str(format("ERROR: unable to write output image to %s") % outpath));
}

// the correct task corrects images
void correct_task(WorkState* state, string inpath, double alt, double pitch, double roll, string outpath) {
  using cv::Mat;
  try {
    Params* params = &state->params;
    log("START CORRECT %s %.2f,%.2f,%.2f") % inpath % alt % pitch % roll;
    // construct the outpath, but throw exception if it exists and skip_existing is true
    outpath = check_outpath(params, inpath, outpath);
    // read RAW image
    Mat cfa_LR = read_16u(inpath);
    // if altitude is out of range, compute from parallax
    alt = compute_missing_alt(state, alt, cfa_LR, inpath);
    // now correct the image
    Mat rgb_LR = correct_one(state, cfa_LR, inpath, alt, pitch, roll);
    // now write corrected image to outpath
    write_corrected(params, rgb_LR, outpath);
    log("CORRECTED %s") % inpath;
  } catch(std::runtime_error const &e) {
    log_error("DID NOT CORRECT %s: %s") % inpath % e.what();
  } catch(std::exception) {
    log_error("ERROR correcting %s") % inpath;
  }
}

// the adaptive task learns and corrects simultaneously
void adaptive_task(WorkState* state, string inpath, double alt, double pitch, double roll, string outpath) {
  using cv::Mat;
  log("START ADAPTIVE %s %.2f") % inpath % alt;
  Mat cfa_LR;
  try {
    // read RAW image
    cfa_LR = read_16u(inpath);
    log("READ %s") % inpath;
  } catch(std::exception) {
    log_error("ERROR failed to read %s") % inpath;
    return; // can't do anything if we can't read the image
  }
  // get a good altitude
  try {
    // if altitude is out of range, compute from parallax
    alt = compute_missing_alt(state, alt, cfa_LR, inpath);
  } catch(std::exception) {
    log_error("NO ALTITUDE available for %s") % inpath;
    return; // the altitude is known to be bad
  }
  try {
    // then learn the image
    learn_one(state, cfa_LR, inpath, alt, pitch, roll);
    log("LEARNED %s") % inpath;
  } catch(std::runtime_error const &e) {
    log_error("DID NOT LEARN %s: %s") % inpath % e.what();
  } catch(std::exception) {
    log_error("ERROR learning %s") % inpath;
  }
  // correct the image
  try {
    Params* params = &state->params;
    // construct the outpath, but throw exception if it exists and skip_existing is true
    outpath = check_outpath(params, inpath, outpath);
    // now correct the image
    Mat rgb_LR = correct_one(state, cfa_LR, inpath, alt, pitch, roll);
    // now write corrected image to outpath
    write_corrected(params, rgb_LR, outpath);
    log("CORRECTED %s") % inpath;
  } catch(std::runtime_error const &e) {
    log_error("DID NOT CORRECT %s: %s") % inpath % e.what();
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

std::string construct_outpath(learn_correct::Params p, string inpath) {
  using boost::algorithm::replace_first;
  using boost::algorithm::ends_with;
  string outpath;
  if(inpath.empty())
    throw std::runtime_error("cannot construct output pathname from empty input pathname");
  if(!p.path_prefix_in.empty()) {
    outpath = inpath;
    replace_first(outpath, p.path_prefix_in, p.path_prefix_out);
  } else if(!p.path_prefix_out.empty()) {
    outpath = p.path_prefix_out + inpath;
  }
  boost::regex re("\\.tiff?$", boost::regex::icase);
  outpath = regex_replace(outpath,re,".png");
  return outpath;
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
  // open/acquire input stream on CSV data
  std::istream* csv_in = get_input(p);
  log("READY to start processing");
  // now do a chunk of work, checkpoint, and continue
  for(int n_todo = 0; n_todo <= 0;) {
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
    n_todo = p.batch_size; // how many images to process in this batch
    string line;
    int n_learned_then = state.n_learned();
    while(getline(*csv_in,line) && (!learn || n_todo--)) { // read pathames from a file
      try {
	// parse the input line and turn it into a Task object
	Task task = Task(line);
	// check that the task is valid
	task.validate();
	// make sure we've got a good outpath
	string outpath; // may be needed for adaptive
	if(correct) { // if correcting
	  outpath = task.outpath.empty() ? construct_outpath(p, task.inpath) : task.outpath;
	  if(outpath.empty())
	    log_error("unable to construct output path for %s") % task.inpath;
	}
	// regardless of which phase we're in, figure out if we really need to do learn/correct work on this inpath/outpath
	bool should_learn = learn;
	if(should_learn && state.should_skip(task.inpath)) {
	  log("SKIPPING LEARN %s - already in lightmap") % task.inpath;
	  should_learn = false;
	}
	bool should_correct = correct;
	if(should_correct && outpath.empty()) {
	  log_error("no output path for %s") % task.inpath;
	  should_correct = false;
	}
	if(should_correct && p.skip_existing && fs::exists(outpath)) {
	  log("SKIPPING correct %s - output image already exists") % task.inpath;
	  should_correct = false;
	}
	// now queue up necessary work
	if(should_learn && should_correct) {
	  // push an adaptive task on the queue
	  io_service.post(boost::bind(adaptive_task, &state, task.inpath, task.alt, task.pitch, task.roll, outpath));
	  log("QUEUED ADAPTIVE %s") % task.inpath;
	} else if(should_learn) {
	  // push a learn task on the queue
	  io_service.post(boost::bind(learn_task, &state, task.inpath, task.alt, task.pitch, task.roll));
	  log("QUEUED LEARN %s") % task.inpath;
	} else if(should_correct) { 
	  // push a correct task on the queue
	  io_service.post(boost::bind(correct_task, &state, task.inpath, task.alt, task.pitch, task.roll, outpath));
	  log("QUEUED CORRECT %s") % task.inpath;
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
  // all chunks done
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
