#pragma once
#include <string>
#include <boost/format.hpp>
#include <boost/tokenizer.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread.hpp>

#define H2O_ADJUSTMENT 1.2 // hardcoded parallax scaling factor
#define MAX_ALTITUDE 10 // hardcoded maximum altitude; sanity check

#define OPT_INPUT "input" // input CSV file (default stdin)
#define OPT_LIGHTMAP_DIR "lightmap" // lightmap directory
#define OPT_BAYER_PATTERN "bayer" // bayer pattern
#define OPT_N_THREADS "threads" // number of threads
#define OPT_ALT_SPACING "alt-spacing" // altitude spacing (m)
#define OPT_FOCAL_LENGTH "focal-length" // focal length (m)
#define OPT_PIXEL_SEP "pixel-size" // pixel size (m)
#define OPT_TEMPLATE_SIZE "patch-size" // parallax tempalte size (pixels)
#define OPT_SMOOTHING "smooth" // lightmap smoothking kernel size (pixels)
#define OPT_CAMERA_SEP "camera-spacing" // distance between focal points of cameras (meters)
#define OPT_MIN_BRIGHTNESS "min-brightness" // min brightness of lightmap (0-1)
#define OPT_MAX_BRIGHTNESS "max-brightness" // max brightness of lightmap (0-1)

namespace po = boost::program_options;

namespace learn_correct {
  // parameters for entire run
  class Params {
  public:
    std::string bayer_pattern;
    int n_threads;
    std::string input;
    std::string lightmap_dir; // was OUT_DIR
    double alt_spacing;
    double focal_length;
    double pixel_sep;
    double h2o_adjustment; // for parallax
    int parallax_template_size; // PARALLAX_TEMPLATE_SIZE
    int lightmap_smoothing; // size of smoothing kernel for lightmap
    double camera_sep; // stereo camera spacing (meters)
    double min_brightness; // min brightness of lightmap
    double max_brightness; // max brightness of lightmap
    // set from command-line options
    Params(po::variables_map options) {
      using std::string;
      input = options[OPT_INPUT].as<string>();
      bayer_pattern = options[OPT_BAYER_PATTERN].as<string>();
      boost::to_lower(bayer_pattern);
      n_threads = options[OPT_N_THREADS].as<int>();
      if(n_threads <= 0) {
	n_threads = boost::thread::hardware_concurrency();
      }
      lightmap_dir = options[OPT_LIGHTMAP_DIR].as<string>();
      alt_spacing = options[OPT_ALT_SPACING].as<double>();
      focal_length = options[OPT_FOCAL_LENGTH].as<double>();
      pixel_sep = options[OPT_PIXEL_SEP].as<double>();
      parallax_template_size = options[OPT_TEMPLATE_SIZE].as<int>();
      lightmap_smoothing = options[OPT_SMOOTHING].as<int>();
      camera_sep = options[OPT_CAMERA_SEP].as<double>();
      min_brightness = options[OPT_MIN_BRIGHTNESS].as<double>();
      max_brightness = options[OPT_MAX_BRIGHTNESS].as<double>();
    }
    friend std::ostream& operator<<(std::ostream &strm, const Params &p) {
      using std::endl;
      strm << OPT_INPUT << " = " << p.input << endl;
      strm << OPT_BAYER_PATTERN << " = " << p.bayer_pattern << endl;
      strm << OPT_N_THREADS << " = " << p.n_threads << endl;
      strm << OPT_LIGHTMAP_DIR << " = " << p.lightmap_dir << endl;
      strm << OPT_ALT_SPACING << " = " << p.alt_spacing << endl;
      strm << OPT_FOCAL_LENGTH << " = " << p.focal_length << endl;
      strm << OPT_PIXEL_SEP << " = " << p.pixel_sep << endl;
      strm << OPT_TEMPLATE_SIZE << " = " << p.parallax_template_size << endl;
      strm << OPT_SMOOTHING << " = " << p.lightmap_smoothing << endl;
      strm << OPT_CAMERA_SEP << " = " << p.camera_sep << endl;
      strm << OPT_MIN_BRIGHTNESS << " = " << p.min_brightness << endl;
      strm << OPT_MAX_BRIGHTNESS << " = " << p.max_brightness << endl;
      return strm;
    }
  };

  // primary work scripts
  void learn(Params p);
  void correct(Params p);

  // parameters for each task
  class Task {
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
    void config(std::vector<std::string> fields) {
      int f=0;
      inpath = fields.at(f++);
      outpath = fields.at(f++);
      alt = atof(fields.at(f++).c_str());
      double pitch_deg = atof(fields.at(f++).c_str());
      double roll_deg = atof(fields.at(f++).c_str());
      pitch = M_PI * pitch_deg / 180.0; // convert pitch to radiasn
      roll = M_PI * roll_deg / 180.0; // convert roll to radians
    }
  public:
    std::string inpath; // input file path
    std::string outpath; // output file path (ignored for learn phase)
    double alt; // altitude (meters)
    double pitch; // pitch (radians)
    double roll; // roll (radians)
    Task(std::vector<std::string> fields) {
      config(fields);
    }
    Task(std::string line) {
      std::vector<std::string> fields;
      Tokenizer tok(line);
      fields.assign(tok.begin(),tok.end());
      config(fields);
    }
    /**
     * Check task to look for values out of range.
     * Does not check altitude, as out of range values are handled
     * by computing altitude from parallax.
     */
    void validate() {
      using boost::format;
      const double extreme_angle = M_PI * 45.0 / 180.0; // FIXME hardcoded
      if(pitch < -extreme_angle || pitch > extreme_angle)
	throw std::runtime_error(str(format("ERROR pitch out of range: %.2f (radians)") % pitch));
      else if(roll < -extreme_angle || roll > extreme_angle)
	throw std::runtime_error(str(format("ERROR roll out of range: %.2f (radians)") % roll));
    }
  };
}
