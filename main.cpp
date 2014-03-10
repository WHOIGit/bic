#include <string>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include "stereo.hpp"
#include "learn_correct.hpp"
#include "prototype.hpp"

using namespace std;

namespace po = boost::program_options;

int main(int argc, char **argv) {
  // non-positional command line options
  po::options_description copts("Command line options");
  copts.add_options()
    ("help,h","produce help message")
    ("command,c",po::value<string>(),"command to run");
  // positional command line options
  po::positional_options_description popts;
  popts.add("command",-1);
  //
  po::variables_map options;
  // parse command line
  po::store(po::command_line_parser(argc, argv).
	    options(copts).positional(popts).run(), options);
  // take action
  string command = options["command"].as<string>();
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
