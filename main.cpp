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
    prototype::test_alt_pitch_roll();
  } else {
    stereo::xoff_test(argc,argv);
  }
}
