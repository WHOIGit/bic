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
  // non-positional command line options
  po::options_description copts("Command line options");
  copts.add_options()
    ("help,h","produce help message")
    (OPT_ABBREV(OPT_COMMAND,"c"),po::value<string>(),"command to run")
    (OPT_ABBREV(OPT_LIGHTMAP_DIR,"l"),po::value<string>(),"lightmap directory")
    (OPT_ABBREV(OPT_BAYER_PATTERN,"b"),po::value<string>(),"bayer pattern (e.g., rggb)")
    (OPT_ABBREV(OPT_N_THREADS,"t"),po::value<int>(),"number of threads")
    (OPT_ABBREV(OPT_ALT_SPACING,"a"),po::value<double>(),"distance between altitude bins (meters)")
    (OPT_ABBREV(OPT_FOCAL_LENGTH,"f"),po::value<double>(),"effective focal length (meters)")
    (OPT_ABBREV(OPT_PIXEL_SEP,"p"),po::value<double>(),"grid spacing of pixels on sensor (meters)")
    (OPT_ABBREV(OPT_TEMPLATE_SIZE,"T"),po::value<int>(),"size of parallax matching template (pixels)")
    (OPT_ABBREV(OPT_SMOOTHING,"s"),po::value<int>(),"size of lightmap smoothing kernel (pixels)")
    ;
  // positional command line options
  po::positional_options_description popts;
  popts.add(OPT_COMMAND,-1);
  // parse command line
  po::parsed_options parsed = po::command_line_parser(argc, argv).options(copts).positional(popts).run();
  po::variables_map options;
  po::store(parsed, options);
  po::notify(options);
  // take action
  if(options.count(OPT_COMMAND)) {
    string command = options[OPT_COMMAND].as<string>();
    cerr << "command is " << command << endl;
    if(command=="learn") {
      learn_correct::learn();
    } else if(command=="correct") {
      learn_correct::correct();
    } else if(command=="res") {
      prototype::test_effective_resolution();
    } else if(command=="flat") {
      prototype::test_flatness();
    } else {
      stereo::xoff_test(argc,argv);
    }
  }
}
