#include <string>
#include "stereo.hpp"
#include "prototype.hpp"

using namespace std;

int main(int argc, char **argv) {
  string command = string(argv[1]);
  if(command=="learn") {
    learn_prototype();
  } else if(command=="correct") {
    correct_prototype();
  } else {
    xoff_test(argc,argv);
  }
}
