#include <string>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include "stereo.hpp"
#include "learn_correct.hpp"
#include "prototype.hpp"

using namespace std;

namespace po = boost::program_options;

#define OPT_COMMAND "command"

#define OPT_ABBREV(x,y) (std::string(x)+","+y).c_str()

int main(int argc, char **argv) {
  using learn_correct::Params;
  // non-positional command line options
  po::options_description copts("Command line options");
  copts.add_options()
    ("help,h","produce help message")
    (OPT_ABBREV(OPT_COMMAND,"c"),po::value<string>(),"command to run")
    (OPT_ABBREV(OPT_LIGHTMAP_DIR,"l"),po::value<string>()->default_value("/tmp"),"lightmap directory")
    (OPT_ABBREV(OPT_BAYER_PATTERN,"b"),po::value<string>()->default_value("rggb"),"bayer pattern (e.g., rggb)")
    (OPT_ABBREV(OPT_N_THREADS,"t"),po::value<int>()->default_value(0),"number of threads")
    (OPT_ABBREV(OPT_ALT_SPACING,"a"),po::value<double>()->default_value(0.1),"distance between altitude bins (meters)")
    (OPT_ABBREV(OPT_FOCAL_LENGTH,"f"),po::value<double>()->default_value(0.012),"effective focal length (meters)")
    (OPT_ABBREV(OPT_PIXEL_SEP,"p"),po::value<double>()->default_value(0.00000645),"grid spacing of pixels on sensor (meters)")
    (OPT_ABBREV(OPT_TEMPLATE_SIZE,"P"),po::value<int>()->default_value(64),"size of parallax matching template (pixels)")
    (OPT_ABBREV(OPT_SMOOTHING,"s"),po::value<int>()->default_value(31),"size of lightmap smoothing kernel (pixels)")
    (OPT_ABBREV(OPT_CAMERA_SEP,"C"),po::value<double>()->default_value(0.235),"stereo camera spacing (meters)")
    (OPT_ABBREV(OPT_MIN_BRIGHTNESS,"m"),po::value<double>()->default_value(0.05),"minimum brightness of lightmap (0-1)")
    (OPT_ABBREV(OPT_MAX_BRIGHTNESS,"M"),po::value<double>()->default_value(0.7),"maximum brightness of lightmap (0-1)")
    (OPT_ABBREV(OPT_INPUT,"i"),po::value<string>()->default_value("-"),"input file (default stdin, or use - to indicate stdin)")
    (OPT_ABBREV(OPT_CREATE_DIRECTORIES,"d"),po::value<bool>()->default_value(true),"create output directories if they don't exist")
    (OPT_ABBREV(OPT_STEREO,"S"),po::value<bool>()->default_value(true),"treat images as side-by-side stereo pairs")
    (OPT_ABBREV(OPT_UPDATE,"u"),po::value<bool>()->default_value(false),"update existing lightmap instead of creating new one")
    (OPT_ABBREV(OPT_ALT_FROM_PARALLAX,"A"),po::value<bool>()->default_value(false),"ignore altitude in input metadata and compute from parallax")
    ;
  po::variables_map options;
  Params params;
  try {
    // positional command line options
    po::positional_options_description popts;
    popts.add(OPT_COMMAND,-1);
    // parse command line
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(copts).positional(popts).run();
    po::store(parsed, options);
    po::notify(options);
    // if the user just wants help, emit usage and exit
    if(options.count("help")) {
      cerr << copts << endl;
      return -1;
    }
    // now set up parameters object
    params = Params(options,false);
    cerr << params;
    params.validate();
  } catch(std::logic_error const &e) {
    cerr << "error: invalid parameter: " << e.what() << endl;
    cerr << copts << endl;
    return -1;
  }
  // take action
  if(options.count(OPT_COMMAND)) {
    string command = options[OPT_COMMAND].as<string>();
    try {
      if(command=="learn") {
	learn_correct::learn(params);
      } else if(command=="correct") {
	learn_correct::correct(params);
      } else if(command=="adaptive") {
	learn_correct::adaptive(params);
      } else if(command=="res") {
	prototype::test_effective_resolution(params);
      } else if(command=="flat") {
	prototype::test_flatness(params);
      } else {
	stereo::xoff_test(argc,argv);
      }
    } catch(std::runtime_error const &e) {
      cerr << "ERROR " << command << ": " << e.what() << endl;
    } catch(std::exception) {
      cerr << "ERROR " << command << endl;
    }
  } else {
    cerr << "Error: no command specified";
  }
}
