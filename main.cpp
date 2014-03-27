#include <string>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include "stereo.hpp"
#include "learn_correct.hpp"
#include "prototype.hpp"
#include "utils.hpp"

using namespace std;

namespace po = boost::program_options;

#define OPT_COMMAND "command"

#define OPT(opt,abbrev,typ,def_val,doc) ((std::string(opt)+","+abbrev).c_str(),po::value<typ>()->default_value(def_val),doc)

int main(int argc, char **argv) {
  using learn_correct::Params;
  // non-positional command line options
  po::options_description copts("Command line options");
  copts.add_options()
    OPT(OPT_COMMAND,"c",string,"command","command to run")
    OPT(OPT_LIGHTMAP_DIR,"l",string,"/tmp","lightmap directory")
    OPT(OPT_BAYER_PATTERN,"b",string,"rggb","bayer pattern (e.g., rggb)")
    OPT(OPT_N_THREADS,"t",int,0,"number of threads")
    OPT(OPT_ALT_SPACING,"a",double,0.1,"distance between altitude bins (meters)")
    OPT(OPT_FOCAL_LENGTH,"f",double,0.012,"effective focal length (meters)")
    OPT(OPT_PIXEL_SEP,"p",double,0.00000645,"grid spacing of pixels on sensor (meters)")
    OPT(OPT_TEMPLATE_SIZE,"P",int,64,"size of parallax matching template (pixels)")
    OPT(OPT_SMOOTHING,"s",int,31,"size of lightmap smoothing kernel (pixels)")
    OPT(OPT_CAMERA_SEP,"C",double,0.235,"stereo camera spacing (meters)")
    OPT(OPT_MIN_BRIGHTNESS,"m",double,0.05,"minimum brightness of lightmap (0-1)")
    OPT(OPT_MAX_BRIGHTNESS,"M",double,0.7,"maximum brightness of lightmap (0-1)")
    OPT(OPT_INPUT,"i",string,"-","input file (default stdin, or use - to indicate stdin)")
    OPT(OPT_CREATE_DIRECTORIES,"d",bool,true,"create output directories if they don't exist")
    OPT(OPT_STEREO,"S",bool,true,"treat images as side-by-side stereo pairs")
    OPT(OPT_UPDATE,"u",string,"","update existing lightmap instead of creating new one (default: don't update a lightmap)")
    OPT(OPT_ALT_FROM_PARALLAX,"A",bool,false,"ignore altitude in input metadata and compute from parallax")
    OPT(OPT_BATCH_SIZE,"B",int,65535,"number of images to learn between checkpoints (learn phase)")
    OPT(OPT_SKIP_EXISTING,"n",bool,false,"skip existing images (correct phase)")
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
      } else if(command=="flat") {
	prototype::test_flatness(params);
      } else if(command=="alt") {
	utils::alt_from_stereo(params);
      } else if(command=="xoff") {
	stereo::xoff_test(argc,argv);
      } else if(command=="res") {
	prototype::test_effective_resolution(params);
      } else if(command=="help") {
	cout << copts << endl;
      } else if(command=="test_dm") {
	prototype::test_dm();
      } else {
	cerr << "ERROR unknown command " << command << endl;
      }
    } catch(std::runtime_error const &e) {
      cerr << "ERROR " << command << ": " << e.what() << endl;
    } catch(std::exception) {
      cerr << "ERROR " << command << endl;
    }
  } else {
    cerr << "ERROR no command specified" << endl;
    cerr << copts << endl;
  }
}
