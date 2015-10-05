#pragma once
#include <string>
#include <boost/format.hpp>
#include <boost/tokenizer.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

#define H2O_ADJUSTMENT 1.2 // hardcoded parallax scaling factor (FIXME magic number?)
#define MAX_ALTITUDE 10 // hardcoded maximum altitude; sanity check

// program option names. these are dynamically associated with
// types, abbreviations, and default values in main.cpp
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
#define OPT_MIN_BRIGHTNESS "min-brightness" // min brightness of lightmap (dimensionless)
#define OPT_MAX_BRIGHTNESS "max-brightness" // max brightness of lightmap (dimensionless)
#define OPT_CREATE_DIRECTORIES "create-directories" // whether to create nonexistent output directories
#define OPT_STEREO "stereo" // whether images are stereo pairs
#define OPT_UPDATE "update" // whether to load lightmap
#define OPT_ALT_FROM_PARALLAX "alt-parallax" // whether to compute altitude from parallax
#define OPT_BATCH_SIZE "batch" // how many images to learn between checkpointing the lightmap
#define OPT_SKIP_EXISTING "new" // whether to skip existing images
#define OPT_PATH_PREFIX_IN "in_prefix" // strip and/or replace input path prefix to construct output path
#define OPT_PATH_PREFIX_OUT "out_prefix" // strip and/or replace input path prefix to construct output path
#define OPT_UNDERTRAIN "undertrain" // undertraining threshold in numbers of images
#define OPT_OVERTRAIN "overtrain" // overtraining threshold in numbers of images
#define OPT_RESOLUTION "resolution" // resolution for some operations (e.g., "1920x1080")
#define OPT_COLOR "color" // are the input images color
#define OPT_CALIBRATION_DIR "calibration" // camera calibration directory for pointcloud stereo
#define OPT_POINTCLOUD_PREFIX "pointcloud_prefix" // strip or replace input path prefix to construct pointcloud prefix

namespace po = boost::program_options;

/**
 * Complete application for learning lightmaps and correction
 * illumination for side-by-side RAW stereo pair TIFFs from a towed
 * benthic vehicle (HabCam V4). Several assumptions are made by this
 * application:
 *
 * - input images are RAW images in 16-bit TIFF format, either single-frame or stereo pairs
 * - desired output is color and illumination-corrected images in 8-bit PNG format
 * - input images are in locally-accessible storage
 * - output images will be written to a locally-accessible storage
 *
 * The application is run in two phases; learn and correct.
 * 
 * In the learn phase, images are averaged together and a lightmap is
 * produced.  The lightmap is stored as a series of 16-bit TIFFs
 * containing statistical information.  Because of the lightmap
 * format, a maximum of 65535 images can be averaged for each altitude
 * bin (see other documentation for details about altitude bins).
 *
 * In the correct phase, images are adjusted according to a
 * previously-learned lightmap to produce color, illumination, and
 * contrast-adjusted color stereo pairs.
 *
 * @author Joe Futrelle
 *
 * @brief application for supervised illumination correction of stereo
 * benthic images
 */
namespace learn_correct {
  /**
   * Configuration parameters. These represent all application
   * paramters the "command" option specifying which procedure
   * the main executable should do (e.g., learn, correct, etc.)
   */
  class Params {
  public:
    /** Bayer pattern of raw images (e.g., rggb), or 'rgb' for color. Case-insensitive */
    std::string bayer_pattern;
    /** Number of threads to use when processing (default: number of CPUs on host) */
    int n_threads;
    /** Input CSV file name (or "-" for stdin) */
    std::string input;
    /** Directory to read/write lightmap to */
    std::string lightmap_dir;
    /** Spacing of altitude bins in meters (default 10cm) */
    double alt_spacing;
    /** Effective focal length of cameras in meters (default 12mm) */
    double focal_length;
    /** Distance between pixel centers on sensor (m) (default 6.45um) */
    double pixel_sep;
    //double h2o_adjustment; // for parallax
    /** For stereo alignment, size of sample template in pixels (default 64) */
    int parallax_template_size; // PARALLAX_TEMPLATE_SIZE
    /** For lightmap smoothing, size of smoothing kernel in pixels (default 31) */
    int lightmap_smoothing; // size of smoothing kernel for lightmap
    /** Distance between cameras in meters (default 23.5cm) */
    double camera_sep; // stereo camera spacing (meters)
    /** Minimum brightness of lightmaps in range (0-1) (default 0.05) */
    double min_brightness; // min brightness of lightmap
    /** Maximum brightness of lightmap in range (0-1) (default 0.7) */
    double max_brightness; // max brightness of lightmap
    /** Whether to create output directories if they do not exist */
    bool create_directories;
    /** Whether to treat images as side-by-side stereo pairs */
    bool stereo;
    /** Which lightmap to load prior to learning */
    std::string update;
    /** Whether to compute altitude from parallax */
    bool alt_from_parallax;
    /** Number of images to learn between checkpoints (learn phase) */
    int batch_size;
    /** Whether to skip processing existing images (correct phase) */
    bool skip_existing;
    /** Path prefix to strip (if any) */
    std::string path_prefix_in;
    /** Replacement path prefix (if any) */
    std::string path_prefix_out;
    /** Undertraining threshold (in numbers of images) */
    int undertrain;
    /** Overtraining threshold (in numbers of images) */
    int overtrain;
    /** X resolution */
    int resolution_x;
    /** Y resolution */
    int resolution_y;
    /** color */
    bool color;
    /** dir for calibration matrices */
    std::string calibration_dir;
    /** prefix for pointcloud output */
    std::string pointcloud_prefix;

    /**
     * Validate parameters. Checks for obviously invalid parameters
     * such as negative focal lengths, min_brightness > max_brightness,
     * etc. When a parameter is wrong, this will throw std::logic_error.
     *
     * In some cases this will produce warnings other than exceptions,
     * e.g., n_threads greater than number of CPUs reported by
     * boost::thread::hardware_concurrency.
     */
    void validate() {
      using namespace std;
      if(!(bayer_pattern == "rggb" || bayer_pattern == "bggr" ||
	   bayer_pattern == "grbg" || bayer_pattern == "gbrg")) {
	throw std::logic_error("unrecognized bayer pattern");
      }
      if(n_threads > boost::thread::hardware_concurrency())
	cerr << "warning: number of threads greater than known number of hardware threads" << endl;
      if(alt_spacing <= 0)
	throw std::logic_error("alt spacing must be > 0m");
      if(alt_spacing > 1)
	cerr << "warning: alt spacing of " << alt_spacing << "m is >1m" << endl;
      if(focal_length <= 0)
	throw std::logic_error("focal length must be > 0m");
      if(focal_length > 1)
	cerr << "warning: focal length of " << focal_length << "m is >1m" << endl;
      if(pixel_sep <= 0)
	throw std::logic_error("pixel separation must be > 0m");
      if(pixel_sep > 0.001)
	cerr << "warning: pixel separation of " << pixel_sep << "m is >1mm" << endl;
      if(parallax_template_size <= 0)
	throw std::logic_error("parallax template size must be > 0px");
      if(parallax_template_size < 8)
	cerr << "warning: parallax template size of " << parallax_template_size << " is likely too small" << endl;
      if(lightmap_smoothing <= 0)
	throw std::logic_error("lightmap smoothing must be > 0px");
      if(camera_sep <= 0)
	throw std::logic_error("camera separation must be > 0m");
      if(min_brightness < 0 || min_brightness > 1)
	throw std::logic_error("min brightness must be between 0 and 1");
      if(max_brightness < 0 || max_brightness > 1)
	throw std::logic_error("max brightness must be between 0 and 1");
      if(min_brightness > max_brightness)
	throw std::logic_error("min brightness is < max brightness");
      if(!stereo && alt_from_parallax)
	throw std::logic_error("cannot compute altitude from parallax without stereo pairs");
      if(batch_size < 0)
	throw std::logic_error("batch size negative");
      if(batch_size < n_threads)
	cerr << "warning: batch size " << batch_size << " less than thread count " << n_threads << endl;
      if(batch_size < 200)
	cerr << "warning: small batch size of " << batch_size << " will result in poor performance in learn phase" << endl;
      if(!path_prefix_in.empty() && path_prefix_out.empty())
	cerr << "warning: input path prefix specified, but output path prefix not specified (use -O to specify)" << endl;
      if(undertrain < 1)
	throw std::logic_error("undertraining threshold must be >= 1 images");
      if(overtrain <= undertrain)
	throw std::logic_error("overtraining threshold must be greater than undertraining threshold");
      if(resolution_x < 0 || resolution_y < 0)
	throw std::logic_error("syntax error in output resolution");
    }
    Params() { }
    void parse_resolution(std::string res) {
      // resolution parameter is bad until proven otherwise
      resolution_x = -1;
      resolution_y = -1;
      //
      if(res.empty())
	return;
      // resolution is in the form "%dx%d"
      try {
	std::vector<std::string> xy;
	boost::split(xy, res, boost::is_any_of("x:")); // allow colons as well
	if(xy.size() != 2)
	  throw std::exception();
	resolution_x = boost::lexical_cast<int>(xy[0]);
	resolution_y = boost::lexical_cast<int>(xy[1]);
      } catch(std::exception) {
	throw std::logic_error(boost::str(boost::format("resolution parameter \"%s\" malformed") % res));
      }
    }
    /**
     * Initialize parameters from a variable map.
     *
     * @param options the variable map
     * @param _validate whether to validate them
     */
    Params(po::variables_map options, bool _validate=true) {
      using std::string;
      input = options[OPT_INPUT].as<string>();
      bayer_pattern = options[OPT_BAYER_PATTERN].as<string>();
      boost::to_lower(bayer_pattern);
      color = bayer_pattern == "rgb";
      n_threads = options[OPT_N_THREADS].as<int>();
      if(n_threads <= 0) { // if number of threads is not specified, find it out
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
      create_directories = options[OPT_CREATE_DIRECTORIES].as<bool>();
      stereo = options[OPT_STEREO].as<bool>();
      update = options[OPT_UPDATE].as<string>();
      alt_from_parallax = options[OPT_ALT_FROM_PARALLAX].as<bool>();
      batch_size = options[OPT_BATCH_SIZE].as<int>();
      skip_existing = options[OPT_SKIP_EXISTING].as<bool>();
      path_prefix_in = options[OPT_PATH_PREFIX_IN].as<string>();
      path_prefix_out = options[OPT_PATH_PREFIX_OUT].as<string>();
      undertrain = options[OPT_UNDERTRAIN].as<int>();
      overtrain = options[OPT_OVERTRAIN].as<int>();
      parse_resolution(options[OPT_RESOLUTION].as<string>());
      calibration_dir = options[OPT_CALIBRATION_DIR].as<string>();
      pointcloud_prefix = options[OPT_POINTCLOUD_PREFIX].as<string>();
      if(_validate)
	validate();
    }
    /**
     * Write a representation of the parameters to an output stream.
     *
     * Options are separated from values by '='
     *
     * @param strm the output stream
     * @param p the parameters
     */
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
      strm << OPT_CREATE_DIRECTORIES << " = " << p.create_directories << endl;
      strm << OPT_STEREO << " = " << p.stereo << endl;
      strm << OPT_UPDATE << " = " << p.update << endl;
      strm << OPT_ALT_FROM_PARALLAX << " = " << p.alt_from_parallax << endl;
      strm << OPT_BATCH_SIZE << " = " << p.batch_size << endl;
      strm << OPT_SKIP_EXISTING << " = " << p.skip_existing << endl;
      strm << OPT_PATH_PREFIX_IN << " = " << p.path_prefix_in << endl;
      strm << OPT_PATH_PREFIX_OUT << " = " << p.path_prefix_out << endl;
      strm << OPT_UNDERTRAIN << " = " << p.undertrain << endl;
      strm << OPT_OVERTRAIN << " = " << p.overtrain << endl;
      strm << OPT_RESOLUTION << p.resolution_x << "x" << p.resolution_y << endl;
      strm << OPT_CALIBRATION_DIR << p.calibration_dir << endl;
      strm << OPT_POINTCLOUD_PREFIX << p.pointcloud_prefix << endl;
      return strm;
    }
  };

  /**
   * Construct outpath from inpath
   */
  std::string construct_path(std::string in_prefix, std::string out_prefix, std::string inpath);
  std::string construct_outpath(Params p, std::string inpath);
  std::string construct_pointcloud_path(Params p, std::string inpath);

  // primary work scripts

  /**
   * Given application parameters, execute the learn phase.
   *
   * @param p the application parameters
   */
  void learn(Params p);
  /**
   * Given application parameters, execute the correct phase.
   *
   * @param p the application parameters
   */
  void correct(Params p);
  /**
   * Given application parameters, execute the adaptive, combined
   * learn/correct phase.
   */
  void adaptive(Params p);
  /**
   * Given application parameters, compute and log altitudes only
   */
  void alt(Params p);
  
  // parameters for each task
  class Task {
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
    void config(std::vector<std::string> fields) {
      // set all fields to default
      alt = 0;
      pitch = 0;
      roll = 0;
      // now process fields
      int f=0;
      // first field is inpath. not optional
      inpath = fields.at(f++);
      try {
	// second field is outpath, but that's optional in learn phase,
	// and so it could be altitude.
	alt = atof(fields.at(f).c_str());
	if(alt != 0) { // we got a nonzero altitude
	  f++; // advance, expecting an outpath followed by altitude
	} else {
	  outpath = fields.at(f++);
	  // if we've reached this point, then there's an outpath so
	  // the next field is altitude.
	  alt = atof(fields.at(f++).c_str());
	}
	double pitch_deg = atof(fields.at(f++).c_str());
	double roll_deg = atof(fields.at(f++).c_str());
	pitch = M_PI * pitch_deg / 180.0; // convert pitch to radiasn
	roll = M_PI * roll_deg / 180.0; // convert roll to radians
      } catch(std::out_of_range) {
	// assume defaults for all missing fields, no need to log this
      }
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
      boost::algorithm::trim_right(line); // deal with \r\n
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
	throw std::runtime_error(str(format("pitch out of range: %.2f (radians)") % pitch));
      else if(roll < -extreme_angle || roll > extreme_angle)
	throw std::runtime_error(str(format("roll out of range: %.2f (radians)") % roll));
    }
  };

  // utilities

  /**
   * Given program options, return a reference to the input stream
   * @param Params the parameters
   */
  std::istream* get_input(Params);
}
