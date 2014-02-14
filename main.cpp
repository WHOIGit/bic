#include <string>
#include "stereo.hpp"
#include "prototype.hpp"

using namespace std;

int main(int argc, char **argv) {
  string command = string(argv[1]);
  if(command=="learn") {
    prototype::learn();
  } else if(command=="correct") {
    prototype::correct();
  } else if(command=="apr") {
    prototype::test_effective_resolution();
  } else if(command=="flat") {
    prototype::test_flatness();
  } else {
    stereo::xoff_test(argc,argv);
  }
}
